//! One byte budget for optional page transfers through a store.

use std::collections::{BTreeMap, HashMap};
use std::net::IpAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::peer_plan::{BudgetEpoch, Membership};
use super::{Key, Writer};

const NANOS_PER_SECOND: u128 = 1_000_000_000;
const MAX_LEASES: usize = 1024;
const LEASE_LIFETIME: Duration = Duration::from_secs(30);
pub(super) const MAX_PEERS: usize = 128;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Limit {
    pub(super) bytes_per_second: u64,
    pub(super) burst_bytes: u64,
}

impl Limit {
    pub(super) fn valid(self) -> bool {
        self.bytes_per_second > 0 && self.burst_bytes > 0
    }

    pub(super) fn valid_share(self) -> bool {
        self.valid() || (self.bytes_per_second == 0 && self.burst_bytes == 0)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PeerLimit {
    pub(super) read: Limit,
    pub(super) write: Limit,
}

pub(super) struct Checkpoints(Mutex<State>);

#[derive(Clone, Copy)]
enum Direction {
    Read,
    Write,
}

struct ReadLease {
    remaining: u64,
    peers: Option<BTreeMap<IpAddr, u64>>,
    expires: Instant,
}

#[derive(Default, Serialize)]
struct Counters {
    read_bytes: u64,
    write_bytes: u64,
    read_reservations_denied: u64,
    write_reservations_denied: u64,
    read_claims_denied: u64,
    expired_read_bytes: u64,
}

#[derive(Serialize)]
pub(super) struct Snapshot<'a> {
    uptime_ms: u64,
    bytes_per_second: Option<u64>,
    burst_bytes: Option<u64>,
    pending_bytes: u64,
    read_leases: usize,
    peers: BTreeMap<IpAddr, PeerSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peer_budget_plan: Option<&'a Membership>,
    #[serde(skip_serializing_if = "Option::is_none")]
    peer_budget_epoch: Option<&'a BudgetEpoch>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    peer_budget_admission_closed: bool,
    #[serde(flatten)]
    counters: &'a Counters,
}

struct Bucket {
    limit: Option<Limit>,
    credits: u128,
    pending: u64,
    completed: u64,
}

impl Bucket {
    fn new(limit: Option<Limit>) -> Self {
        Self {
            limit,
            credits: u128::from(limit.map_or(0, |item| item.burst_bytes)) * NANOS_PER_SECOND,
            pending: 0,
            completed: 0,
        }
    }

    fn refill(&mut self, elapsed: u128) {
        if let Some(limit) = self.limit {
            let available = u128::from(limit.burst_bytes - self.pending) * NANOS_PER_SECOND;
            let earned = elapsed.saturating_mul(u128::from(limit.bytes_per_second));
            self.credits = self.credits.saturating_add(earned).min(available);
        }
    }

    fn fits(&self, bytes: u64) -> bool {
        self.limit.is_none() || u128::from(bytes) * NANOS_PER_SECOND <= self.credits
    }

    fn charge(&mut self, bytes: u64) {
        if self.limit.is_some() {
            self.credits -= u128::from(bytes) * NANOS_PER_SECOND;
        }
        self.pending += bytes;
    }

    fn finish(&mut self, bytes: u64, complete: bool) {
        self.pending -= bytes;
        if complete {
            self.completed = self.completed.saturating_add(bytes);
        }
    }
}

struct PeerBuckets {
    read: Bucket,
    write: Bucket,
    denied_reads: u64,
    denied_writes: u64,
}

impl PeerBuckets {
    fn new(limit: PeerLimit) -> Self {
        Self {
            read: Bucket::new(Some(limit.read)),
            write: Bucket::new(Some(limit.write)),
            denied_reads: 0,
            denied_writes: 0,
        }
    }
}

#[derive(Serialize)]
struct PeerSnapshot {
    read: Limit,
    write: Limit,
    pending_read_bytes: u64,
    pending_write_bytes: u64,
    read_bytes: u64,
    write_bytes: u64,
    read_reservations_denied: u64,
    write_reservations_denied: u64,
}

struct State {
    global: Bucket,
    peers: BTreeMap<IpAddr, PeerBuckets>,
    plan: Option<Membership>,
    epoch: Option<BudgetEpoch>,
    admission_closed: bool,
    started: Instant,
    updated: Instant,
    reads: HashMap<(Key, Writer), ReadLease>,
    counters: Counters,
}

impl State {
    fn peers_are_restricted(&self) -> bool {
        self.plan.is_some() || !self.peers.is_empty()
    }

    fn refresh(&mut self, now: Instant) {
        let now = now.max(self.updated);
        let elapsed = now.duration_since(self.updated).as_nanos();
        self.global.refill(elapsed);
        for peer in self.peers.values_mut() {
            peer.read.refill(elapsed);
            peer.write.refill(elapsed);
        }
        self.updated = now;
        self.reads.retain(|_, lease| {
            if lease.expires > now {
                return true;
            }
            self.global.finish(lease.remaining, false);
            if !self.peers.is_empty()
                && let Some(peers) = &lease.peers
            {
                for (peer, &bytes) in peers {
                    self.peers
                        .get_mut(peer)
                        .expect("reserved peer")
                        .read
                        .finish(bytes, false);
                }
            }
            self.counters.expired_read_bytes = self
                .counters
                .expired_read_bytes
                .saturating_add(lease.remaining);
            false
        });
    }

    fn fits_read(&self, bytes: u64, peers: Option<&BTreeMap<IpAddr, u64>>) -> bool {
        self.global.fits(bytes)
            && (!self.peers_are_restricted()
                || peers.is_some_and(|peers| {
                    peers.iter().all(|(ip, &bytes)| {
                        self.peers.get(ip).is_some_and(|peer| peer.read.fits(bytes))
                    })
                }))
    }

    fn charge_read(&mut self, bytes: u64, peers: Option<&BTreeMap<IpAddr, u64>>) {
        self.global.charge(bytes);
        if !self.peers.is_empty()
            && let Some(peers) = peers
        {
            for (ip, &bytes) in peers {
                self.peers
                    .get_mut(ip)
                    .expect("checked peer")
                    .read
                    .charge(bytes);
            }
        }
    }
}

fn buckets(
    limit: Option<Limit>,
    peers: BTreeMap<IpAddr, PeerLimit>,
    cold: bool,
) -> (Bucket, BTreeMap<IpAddr, PeerBuckets>) {
    let mut global = Bucket::new(limit);
    let mut peers: BTreeMap<_, _> = peers
        .into_iter()
        .map(|(ip, limit)| (ip, PeerBuckets::new(limit)))
        .collect();
    if cold {
        // Restart and plan activation must earn a burst from fresh credit.
        global.credits = 0;
        for peer in peers.values_mut() {
            peer.read.credits = 0;
            peer.write.credits = 0;
        }
    }
    (global, peers)
}

impl Checkpoints {
    pub(super) fn new(
        limit: Option<Limit>,
        peers: BTreeMap<IpAddr, PeerLimit>,
        now: Instant,
        plan: Option<Membership>,
    ) -> Self {
        let (global, peers) = buckets(limit, peers, plan.is_some());
        Self(Mutex::new(State {
            global,
            peers,
            plan,
            epoch: None,
            admission_closed: false,
            started: now,
            updated: now,
            reads: HashMap::new(),
            counters: Counters::default(),
        }))
    }

    pub(super) fn set_epoch(&self, epoch: BudgetEpoch) {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.admission_closed = epoch.target_plan_id.is_some();
        state.epoch = Some(epoch);
    }

    pub(super) fn close_admission(&self) {
        self.0.lock().expect("checkpoint budget").admission_closed = true;
    }

    pub(super) fn mark_epoch_uncertain(&self) {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.admission_closed = true;
        state.epoch = None;
    }

    pub(super) fn drained(&self, now: Instant) -> bool {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        state.admission_closed && state.global.pending == 0
    }

    pub(super) fn install_plan(
        &self,
        plan: Membership,
        peers: BTreeMap<IpAddr, PeerLimit>,
        epoch: BudgetEpoch,
        now: Instant,
    ) {
        let mut state = self.0.lock().expect("checkpoint budget");
        assert!(state.admission_closed && state.global.pending == 0 && state.reads.is_empty());
        (state.global, state.peers) = buckets(state.global.limit, peers, true);
        state.updated = now.max(state.updated);
        state.plan = Some(plan);
        state.epoch = Some(epoch);
        state.admission_closed = false;
    }

    pub(super) fn reserve_read(
        &self,
        key: Key,
        writer: Writer,
        bytes: u64,
        peers: Option<BTreeMap<IpAddr, u64>>,
        now: Instant,
    ) -> bool {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        let id = (key, writer);
        if state.admission_closed
            || state.reads.len() >= MAX_LEASES
            || state.reads.contains_key(&id)
            || peers.as_ref().is_some_and(|peers| {
                peers
                    .values()
                    .try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
                    != Some(bytes)
            })
            || !state.fits_read(bytes, peers.as_ref())
        {
            if let Some(peers) = &peers {
                for (ip, &bytes) in peers {
                    if let Some(peer) = state
                        .peers
                        .get_mut(ip)
                        .filter(|peer| !peer.read.fits(bytes))
                    {
                        peer.denied_reads = peer.denied_reads.saturating_add(1);
                    }
                }
            }
            state.counters.read_reservations_denied =
                state.counters.read_reservations_denied.saturating_add(1);
            return false;
        }
        state.charge_read(bytes, peers.as_ref());
        let expires = state.updated + LEASE_LIFETIME;
        state.reads.insert(
            id,
            ReadLease {
                remaining: bytes,
                peers,
                expires,
            },
        );
        true
    }

    pub(super) fn read(
        &self,
        key: Key,
        writer: Writer,
        peer: IpAddr,
        bytes: u64,
        now: Instant,
    ) -> Option<Transfer<'_>> {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        let id = (key, writer);
        let Some(lease) = state.reads.get_mut(&id).filter(|lease| {
            lease.remaining >= bytes
                && lease.peers.as_ref().is_none_or(|peers| {
                    peers
                        .get(&peer)
                        .is_some_and(|&remaining| remaining >= bytes)
                })
        }) else {
            state.counters.read_claims_denied = state.counters.read_claims_denied.saturating_add(1);
            return None;
        };
        if let Some(peers) = &mut lease.peers {
            *peers.get_mut(&peer).expect("checked lease peer") -= bytes;
        }
        lease.remaining -= bytes;
        if lease.remaining == 0 {
            state.reads.remove(&id);
        }
        Some(Transfer {
            budget: self,
            direction: Direction::Read,
            peer,
            bytes,
            complete: false,
        })
    }

    pub(super) fn write(&self, peer: IpAddr, bytes: u64, now: Instant) -> Option<Transfer<'_>> {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        if state.admission_closed
            || !state.global.fits(bytes)
            || (state.peers_are_restricted()
                && !state
                    .peers
                    .get(&peer)
                    .is_some_and(|buckets| buckets.write.fits(bytes)))
        {
            if let Some(peer) = state
                .peers
                .get_mut(&peer)
                .filter(|peer| !peer.write.fits(bytes))
            {
                peer.denied_writes = peer.denied_writes.saturating_add(1);
            }
            state.counters.write_reservations_denied =
                state.counters.write_reservations_denied.saturating_add(1);
            return None;
        }
        state.global.charge(bytes);
        if let Some(buckets) = state.peers.get_mut(&peer) {
            buckets.write.charge(bytes);
        }
        Some(Transfer {
            budget: self,
            direction: Direction::Write,
            peer,
            bytes,
            complete: false,
        })
    }

    pub(super) fn snapshot(&self, now: Instant) -> Vec<u8> {
        let mut state = self.0.lock().expect("checkpoint budget");
        state.refresh(now);
        serde_json::to_vec(&Snapshot {
            uptime_ms: state.updated.duration_since(state.started).as_millis() as u64,
            bytes_per_second: state.global.limit.map(|limit| limit.bytes_per_second),
            burst_bytes: state.global.limit.map(|limit| limit.burst_bytes),
            pending_bytes: state.global.pending,
            read_leases: state.reads.len(),
            peers: state
                .peers
                .iter()
                .map(|(&ip, peer)| {
                    (
                        ip,
                        PeerSnapshot {
                            read: peer.read.limit.expect("configured read budget"),
                            write: peer.write.limit.expect("configured write budget"),
                            pending_read_bytes: peer.read.pending,
                            pending_write_bytes: peer.write.pending,
                            read_bytes: peer.read.completed,
                            write_bytes: peer.write.completed,
                            read_reservations_denied: peer.denied_reads,
                            write_reservations_denied: peer.denied_writes,
                        },
                    )
                })
                .collect(),
            peer_budget_plan: state.plan.as_ref(),
            peer_budget_epoch: state.epoch.as_ref(),
            peer_budget_admission_closed: state.admission_closed,
            counters: &state.counters,
        })
        .expect("serializable checkpoint metrics")
    }
}

pub(super) struct Transfer<'a> {
    budget: &'a Checkpoints,
    direction: Direction,
    peer: IpAddr,
    bytes: u64,
    complete: bool,
}

impl Transfer<'_> {
    pub(super) fn complete(mut self) {
        self.complete = true;
    }
}

impl Drop for Transfer<'_> {
    fn drop(&mut self) {
        let mut state = self.budget.0.lock().expect("checkpoint budget");
        state.refresh(Instant::now());
        state.global.finish(self.bytes, false);
        if let Some(peer) = state.peers.get_mut(&self.peer) {
            match self.direction {
                Direction::Read => &mut peer.read,
                Direction::Write => &mut peer.write,
            }
            .finish(self.bytes, self.complete);
        }
        if self.complete {
            let counter = match self.direction {
                Direction::Read => &mut state.counters.read_bytes,
                Direction::Write => &mut state.counters.write_bytes,
            };
            *counter = counter.saturating_add(self.bytes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn local() -> IpAddr {
        Ipv4Addr::LOCALHOST.into()
    }

    #[test]
    fn out_of_order_clock_samples_cannot_refill_the_same_interval_twice() {
        let start = Instant::now();
        let budget = Checkpoints::new(
            Some(Limit {
                bytes_per_second: 10,
                burst_bytes: 20,
            }),
            BTreeMap::new(),
            start,
            None,
        );
        let mut state = budget.0.lock().unwrap();
        state.global.charge(20);
        state.global.finish(20, false);
        state.refresh(start + Duration::from_millis(500));
        state.refresh(start + Duration::from_millis(200));
        state.refresh(start + Duration::from_millis(800));
        assert_eq!(state.global.credits, 8 * NANOS_PER_SECOND);
        assert!(!state.global.fits(9));
    }

    #[test]
    fn outstanding_reads_and_writes_share_one_burst_and_refill() {
        let now = Instant::now();
        let budget = Checkpoints::new(
            Some(Limit {
                bytes_per_second: 100,
                burst_bytes: 100,
            }),
            BTreeMap::new(),
            now,
            None,
        );
        assert!(budget.reserve_read([1; 32], [1; 16], 80, None, now));
        let held = budget.write(local(), 20, now).unwrap();
        let later = now + Duration::from_secs(1);
        assert!(budget.write(local(), 1, later).is_none());
        assert!(budget.read([1; 32], [2; 16], local(), 1, later).is_none());
        let read = budget.read([1; 32], [1; 16], local(), 30, later).unwrap();
        assert!(budget.write(local(), 1, later).is_none());
        read.complete();
        drop(held);
        assert!(budget.write(local(), 1, later).is_none());
        let later = later + Duration::from_millis(500);
        budget.write(local(), 50, later).unwrap().complete();
        assert!(budget.read([1; 32], [1; 16], local(), 51, later).is_none());
        assert!(
            budget
                .read([1; 32], [1; 16], local(), 50, now + LEASE_LIFETIME)
                .is_none()
        );
        let snapshot: serde_json::Value =
            serde_json::from_slice(&budget.snapshot(now + LEASE_LIFETIME)).unwrap();
        assert_eq!(snapshot["pending_bytes"], 0);
        assert_eq!(snapshot["read_bytes"], 30);
        assert_eq!(snapshot["write_bytes"], 50);
        assert_eq!(snapshot["expired_read_bytes"], 50);
    }

    #[test]
    fn a_weak_peer_refuses_the_whole_read_and_never_charges_other_buckets() {
        let now = Instant::now();
        let [a, b, c] = [1, 2, 3].map(|host| IpAddr::V4(Ipv4Addr::new(192, 0, 2, host)));
        let limit = |burst_bytes| Limit {
            bytes_per_second: 20,
            burst_bytes,
        };
        let budget = Checkpoints::new(
            Some(limit(20)),
            BTreeMap::from([
                (
                    a,
                    PeerLimit {
                        read: limit(12),
                        write: limit(3),
                    },
                ),
                (
                    b,
                    PeerLimit {
                        read: limit(4),
                        write: limit(8),
                    },
                ),
            ]),
            now,
            None,
        );
        let key = [1; 32];
        let lease = [2; 16];
        assert!(!budget.reserve_read(key, lease, 16, None, now));
        assert!(!budget.reserve_read(key, lease, 16, Some(BTreeMap::from([(a, 8), (b, 8)])), now));
        assert!(!budget.reserve_read(key, lease, 16, Some(BTreeMap::from([(a, 12), (c, 4)])), now));
        assert!(budget.reserve_read(key, lease, 16, Some(BTreeMap::from([(a, 12), (b, 4)])), now));
        assert!(budget.read(key, lease, c, 1, now).is_none());
        assert!(budget.read(key, lease, b, 5, now).is_none());
        assert!(budget.write(a, 4, now).is_none());
        assert!(budget.write(c, 1, now).is_none());
        budget.write(a, 3, now).unwrap().complete();
        budget.read(key, lease, a, 12, now).unwrap().complete();
        let later = now + LEASE_LIFETIME;
        assert!(budget.read(key, lease, b, 4, later).is_none());
        let stats: serde_json::Value = serde_json::from_slice(&budget.snapshot(later)).unwrap();
        assert_eq!(stats["pending_bytes"], 0);
        assert_eq!(stats["expired_read_bytes"], 4);
        assert_eq!(stats["peers"][a.to_string()]["read_bytes"], 12);
        assert_eq!(stats["peers"][a.to_string()]["write_bytes"], 3);
        assert_eq!(
            stats["peers"][a.to_string()]["write_reservations_denied"],
            1
        );
        assert_eq!(stats["peers"][b.to_string()]["read_reservations_denied"], 1);
        assert_eq!(stats["peers"][b.to_string()]["pending_read_bytes"], 0);
    }
}
