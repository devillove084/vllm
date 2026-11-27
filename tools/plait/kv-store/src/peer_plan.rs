//! Disjoint shares of a worker's network budget across KV stores.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read};
use std::net::{IpAddr, SocketAddr};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::task::JoinSet;

use super::checkpoints::{Limit, MAX_PEERS, PeerLimit};
use super::{DEADLINE, read_stats, unicast_peer};

const MAX_STORES: usize = 128;
pub(super) const MAX_PLAN_BYTES: u64 = 8 << 20;

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Peer {
    address: IpAddr,
    limit: PeerLimit,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Store {
    endpoint: SocketAddr,
    peers: Vec<Peer>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Document {
    peers: Vec<Peer>,
    stores: Vec<Store>,
}

#[derive(Serialize)]
struct Allocation {
    peers: BTreeMap<IpAddr, PeerLimit>,
    stores: BTreeMap<SocketAddr, BTreeMap<IpAddr, PeerLimit>>,
}

pub(super) struct PeerPlan {
    document: Document,
    allocation: Allocation,
    id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub(super) struct Membership {
    id: String,
    endpoint: SocketAddr,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub(super) struct BudgetEpoch {
    pub(super) generation: u64,
    pub(super) previous_plan_id: Option<String>,
    pub(super) target_plan_id: Option<String>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub(super) fn canonical_endpoint(endpoint: SocketAddr) -> SocketAddr {
    match endpoint {
        SocketAddr::V6(value) if value.ip().to_ipv4_mapped().is_some() => {
            SocketAddr::new(endpoint.ip().to_canonical(), endpoint.port())
        }
        _ => endpoint,
    }
}

fn peer_map(entries: &[Peer]) -> io::Result<BTreeMap<IpAddr, PeerLimit>> {
    if entries.len() > MAX_PEERS {
        return Err(invalid("invalid number of peers in checkpoint plan"));
    }
    let mut peers = BTreeMap::new();
    for &Peer { address, limit } in entries {
        let ip = address.to_canonical();
        if !unicast_peer(ip) || !limit.read.valid_share() || !limit.write.valid_share() {
            return Err(invalid(format!("invalid checkpoint budget for peer {ip}")));
        }
        if peers.insert(ip, limit).is_some() {
            return Err(invalid(format!("duplicate checkpoint peer {ip}")));
        }
    }
    Ok(peers)
}

fn debit(capacity: &mut Limit, quota: Limit) -> Option<()> {
    let bytes_per_second = capacity
        .bytes_per_second
        .checked_sub(quota.bytes_per_second)?;
    let burst_bytes = capacity.burst_bytes.checked_sub(quota.burst_bytes)?;
    *capacity = Limit {
        bytes_per_second,
        burst_bytes,
    };
    Some(())
}

impl PeerPlan {
    pub(super) fn load(path: &str) -> io::Result<Self> {
        let mut body = Vec::new();
        File::open(path)?
            .take(MAX_PLAN_BYTES + 1)
            .read_to_end(&mut body)?;
        if body.len() as u64 > MAX_PLAN_BYTES {
            return Err(invalid("checkpoint budget plan is too large"));
        }
        Self::validate(serde_json::from_slice(&body)?)
    }

    pub(super) fn validate(document: Document) -> io::Result<Self> {
        let peers = peer_map(&document.peers)?;
        if peers.is_empty() {
            return Err(invalid("checkpoint plan needs at least one peer budget"));
        }
        if !(1..=MAX_STORES).contains(&document.stores.len()) {
            return Err(invalid("invalid number of stores in checkpoint plan"));
        }
        let mut remaining = peers.clone();
        let mut stores = BTreeMap::new();
        for Store { endpoint, peers } in &document.stores {
            let endpoint = canonical_endpoint(*endpoint);
            if endpoint.port() == 0 || !unicast_peer(endpoint.ip()) {
                return Err(invalid(format!(
                    "invalid checkpoint store endpoint {endpoint}"
                )));
            }
            let peers = peer_map(peers)?;
            if stores.contains_key(&endpoint) {
                return Err(invalid(format!("duplicate checkpoint endpoint {endpoint}")));
            }
            for (&ip, limit) in &peers {
                let capacity = remaining.get_mut(&ip).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint store {endpoint} names undeclared peer {ip}"
                    ))
                })?;
                debit(&mut capacity.read, limit.read).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint read shares exceed the rate or burst for {ip}"
                    ))
                })?;
                debit(&mut capacity.write, limit.write).ok_or_else(|| {
                    invalid(format!(
                        "checkpoint write shares exceed the rate or burst for {ip}"
                    ))
                })?;
            }
            stores.insert(endpoint, peers);
        }
        let allocation = Allocation { peers, stores };
        let mut hash = Sha256::new();
        hash.update(b"plait-checkpoint-peer-plan-v1");
        hash.update(serde_json::to_vec(&allocation)?);
        let id = format!("{:x}", hash.finalize());
        Ok(Self {
            document,
            allocation,
            id,
        })
    }

    pub(super) fn id(&self) -> &str {
        &self.id
    }

    pub(super) fn document(&self) -> &Document {
        &self.document
    }

    pub(super) fn validate_successor(&self, target: &Self) -> io::Result<()> {
        if self.id == target.id {
            return Err(invalid("checkpoint budget successor must change the plan"));
        }
        if self
            .allocation
            .stores
            .keys()
            .ne(target.allocation.stores.keys())
        {
            return Err(invalid(
                "checkpoint budget successor must retain the store endpoints",
            ));
        }
        Ok(())
    }

    pub(super) fn share_for(
        &self,
        endpoint: SocketAddr,
    ) -> io::Result<(Membership, BTreeMap<IpAddr, PeerLimit>)> {
        let endpoint = canonical_endpoint(endpoint);
        let peers = self.allocation.stores.get(&endpoint).ok_or_else(|| {
            invalid(format!(
                "checkpoint budget plan has no allocation for {endpoint}"
            ))
        })?;
        Ok((
            Membership {
                id: self.id.clone(),
                endpoint,
            },
            peers.clone(),
        ))
    }

    fn reports(&self, skip: Option<SocketAddr>) -> JoinSet<io::Result<(SocketAddr, Report)>> {
        let mut jobs = JoinSet::new();
        for &endpoint in self.allocation.stores.keys() {
            if skip == Some(endpoint) {
                continue;
            }
            jobs.spawn(async move {
                let context = |err: &dyn std::error::Error| {
                    io::Error::other(format!("checkpoint store {endpoint}: {err}"))
                };
                let body = tokio::time::timeout(DEADLINE, read_stats(endpoint))
                    .await
                    .map_err(|err| context(&err))?
                    .map_err(|err| context(&err))?;
                let report = serde_json::from_str(&body).map_err(|err| context(&err))?;
                Ok((endpoint, report))
            });
        }
        jobs
    }

    pub(super) async fn require_drained(
        &self,
        local_endpoint: SocketAddr,
        generation: u64,
        target: &Self,
    ) -> io::Result<()> {
        let next = generation
            .checked_add(1)
            .ok_or_else(|| invalid("checkpoint budget generation exhausted"))?;
        let mut jobs = self.reports(Some(canonical_endpoint(local_endpoint)));
        while let Some(result) = jobs.join_next().await {
            let (endpoint, report) = result.map_err(io::Error::other)??;
            let drained = report.peer_budget_epoch.as_ref().is_some_and(|epoch| {
                let matches_plan = |id: &str| {
                    report
                        .peer_budget_plan
                        .as_ref()
                        .is_some_and(|plan| plan.endpoint == endpoint && plan.id == id)
                };
                (epoch.generation == generation
                    && matches_plan(self.id())
                    && epoch.target_plan_id.as_deref() == Some(target.id())
                    && report.peer_budget_admission_closed
                    && report.pending_bytes == Some(0))
                    || (epoch.generation == next
                        && matches_plan(target.id())
                        && epoch.previous_plan_id.as_deref() == Some(self.id()))
            });
            if !drained {
                return Err(invalid(format!(
                    "checkpoint store {endpoint} has not drained generation {generation} for {}",
                    target.id()
                )));
            }
        }
        Ok(())
    }

    pub(super) async fn audit(&self) -> io::Result<String> {
        let mut jobs = self.reports(None);
        let mut expected_generation = None;
        while let Some(result) = jobs.join_next().await {
            let (endpoint, report) = result.map_err(io::Error::other)??;
            let generation = report
                .peer_budget_epoch
                .as_ref()
                .map(|epoch| epoch.generation);
            if report.peer_budget_admission_closed
                || report
                    .peer_budget_epoch
                    .is_some_and(|epoch| epoch.target_plan_id.is_some())
                || *expected_generation.get_or_insert(generation) != generation
            {
                return Err(invalid(format!(
                    "checkpoint store {endpoint} is draining or on a different budget generation"
                )));
            }
            let (expected, peers) = self.share_for(endpoint)?;
            let actual: BTreeMap<_, _> = report
                .peers
                .into_iter()
                .map(|(ip, share)| {
                    (
                        ip,
                        PeerLimit {
                            read: share.read,
                            write: share.write,
                        },
                    )
                })
                .collect();
            if report.peer_budget_plan.as_ref() != Some(&expected) || peers != actual {
                return Err(invalid(format!(
                    "checkpoint store {endpoint} has a different budget plan or share"
                )));
            }
        }
        let mut summary = serde_json::json!({
            "id": self.id,
            "stores": self.allocation.stores.keys().collect::<Vec<_>>(),
        });
        if let Some(Some(generation)) = expected_generation {
            summary["generation"] = generation.into();
        }
        Ok(summary.to_string())
    }
}

#[derive(Deserialize)]
struct Report {
    peer_budget_plan: Option<Membership>,
    peer_budget_epoch: Option<BudgetEpoch>,
    #[serde(default)]
    peer_budget_admission_closed: bool,
    pending_bytes: Option<u64>,
    peers: BTreeMap<IpAddr, ReportedShare>,
}

#[derive(Deserialize)]
struct ReportedShare {
    read: Limit,
    write: Limit,
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use serde_json::{Value, json};

    use super::*;
    use crate::checkpoints::Checkpoints;

    fn fixture() -> Value {
        let limits = |rate, burst| {
            json!({
                "read": {"bytes_per_second": rate, "burst_bytes": burst},
                "write": {"bytes_per_second": rate, "burst_bytes": burst}
            })
        };
        let entries = |rate, burst| {
            json!([{
                "address": "192.0.2.3", "limit": limits(rate, burst)
            }])
        };
        json!({
            "peers": entries(100, 10),
            "stores": [
                {"endpoint": "192.0.2.5:7952", "peers": entries(40, 4)},
                {"endpoint": "192.0.2.6:7952", "peers": entries(60, 6)}
            ]
        })
    }

    fn parse(value: Value) -> io::Result<PeerPlan> {
        PeerPlan::validate(serde_json::from_value(value)?)
    }

    #[test]
    fn same_plan_has_a_stable_id_and_each_endpoint_receives_only_its_share() {
        let plan = parse(fixture()).unwrap();
        let mut reordered = fixture();
        reordered["stores"].as_array_mut().unwrap().reverse();
        reordered["stores"][1]["endpoint"] = json!("[::ffff:192.0.2.5]:7952");
        assert_eq!(plan.id, parse(reordered).unwrap().id);
        let endpoint = "192.0.2.5:7952".parse().unwrap();
        let peer = "192.0.2.3".parse::<IpAddr>().unwrap();
        let (membership, shares) = plan.share_for(endpoint).unwrap();
        assert_eq!(membership.endpoint, endpoint);
        assert_eq!(
            shares[&peer].read,
            Limit {
                bytes_per_second: 40,
                burst_bytes: 4
            }
        );
        assert!(plan.share_for("192.0.2.4:7952".parse().unwrap()).is_err());
    }

    #[test]
    fn rejects_overallocated_rates_bursts_unknown_peers_and_duplicate_endpoints() {
        for field in ["bytes_per_second", "burst_bytes"] {
            let mut invalid = fixture();
            let budget = &mut invalid["stores"][1]["peers"][0]["limit"]["write"][field];
            *budget = json!(budget.as_u64().unwrap() + 1);
            assert!(parse(invalid).is_err());
        }
        let mut unknown = fixture();
        unknown["stores"][1]["peers"][0]["address"] = json!("192.0.2.4");
        assert!(parse(unknown).is_err());

        let mut duplicate = fixture();
        duplicate["stores"][1]["endpoint"] = json!("[::ffff:192.0.2.5]:7952");
        assert!(parse(duplicate).is_err());
    }

    #[test]
    fn restart_accrues_fresh_credit_and_empty_or_zero_shares_stay_restricted() {
        let endpoint = "192.0.2.5:7952".parse().unwrap();
        let other = "192.0.2.6:7952".parse().unwrap();
        let peer = "192.0.2.3".parse().unwrap();
        let start = Instant::now();
        let plan = parse(fixture()).unwrap();
        let (member, shares) = plan.share_for(endpoint).unwrap();
        let first = Checkpoints::new(None, shares.clone(), start, Some(member.clone()));
        assert!(first.write(peer, 1, start).is_none());
        let restart = start + Duration::from_millis(100);
        first.write(peer, 4, restart).unwrap().complete();
        let second = Checkpoints::new(None, shares, restart, Some(member));
        assert!(second.write(peer, 1, restart).is_none());
        second
            .write(peer, 4, restart + Duration::from_millis(100))
            .unwrap()
            .complete();

        let mut disabled = fixture();
        disabled["stores"][0]["peers"][0]["limit"]["read"] =
            json!({"bytes_per_second": 0, "burst_bytes": 0});
        disabled["stores"][1]["peers"] = json!([]);
        let plan = parse(disabled).unwrap();
        let late = start + Duration::from_secs(30);
        let (member, shares) = plan.share_for(endpoint).unwrap();
        let store = Checkpoints::new(None, shares, start, Some(member));
        assert!(!store.reserve_read([1; 32], [2; 16], 1, Some(BTreeMap::from([(peer, 1)])), late));
        let (member, shares) = plan.share_for(other).unwrap();
        let store = Checkpoints::new(None, shares, start, Some(member));
        assert!(store.write(peer, 1, late).is_none());
        assert!(!store.reserve_read([1; 32], [2; 16], 1, None, late));
    }

    #[test]
    fn switch_waits_for_grants_and_does_not_transfer_old_credit() {
        let endpoint = "192.0.2.5:7952".parse().unwrap();
        let peer = "192.0.2.3".parse().unwrap();
        let start = Instant::now();
        let old = parse(fixture()).unwrap();
        let mut data = fixture();
        data["stores"].as_array_mut().unwrap().rotate_left(1);
        data["stores"][0]["endpoint"] = json!("192.0.2.5:7952");
        data["stores"][1]["endpoint"] = json!("192.0.2.6:7952");
        let new = parse(data).unwrap();
        let (member, shares) = old.share_for(endpoint).unwrap();
        let store = Checkpoints::new(None, shares, start, Some(member));
        let ready = start + Duration::from_secs(1);
        let key = [1; 32];
        let writer = [2; 16];
        assert!(store.reserve_read(key, writer, 4, Some(BTreeMap::from([(peer, 4)])), ready));
        let upload = store.write(peer, 4, ready).unwrap();
        store.set_epoch(BudgetEpoch {
            generation: 0,
            previous_plan_id: None,
            target_plan_id: Some(new.id.clone()),
        });
        assert!(!store.reserve_read(key, [3; 16], 1, Some(BTreeMap::from([(peer, 1)])), ready));
        upload.complete();
        assert!(store.write(peer, 1, ready).is_none());
        assert!(!store.drained(ready));
        let download = store.read(key, writer, peer, 4, ready).unwrap();
        assert!(!store.drained(ready));
        download.complete();
        assert!(store.drained(ready));

        let (member, shares) = new.share_for(endpoint).unwrap();
        let switched = ready + Duration::from_secs(1);
        store.install_plan(
            member,
            shares,
            BudgetEpoch {
                generation: 1,
                previous_plan_id: Some(old.id),
                target_plan_id: None,
            },
            switched,
        );
        assert!(store.write(peer, 1, switched).is_none());
        let later = switched + Duration::from_millis(100);
        store.write(peer, 6, later).unwrap().complete();
        assert!(store.read(key, writer, peer, 1, later).is_none());
    }
}
