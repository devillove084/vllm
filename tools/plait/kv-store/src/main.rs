//! Bounded storage for complete KV blocks in logical head and layer order.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

#[cfg(unix)]
mod budget_control;
mod checkpoints;
mod peer_plan;
use checkpoints::{Checkpoints, Limit, MAX_PEERS, PeerLimit};
use peer_plan::PeerPlan;

const MAGIC: &[u8; 8] = b"PLAITKV2";
const RANGED_MAGIC: &[u8; 8] = b"PLAITKV3";
const PEER_READ_MAGIC: &[u8; 8] = b"PLAITKV4";
const METRICS_MAGIC: &[u8; 8] = b"PLAITKVM";
const HEADER: usize = 73;
const PEER_ENTRY: usize = 24;
const MAX_METRICS_BYTES: usize = 128 << 10;
const MAX_BLOCK_BYTES: usize = 64 << 20;
const MAX_HEADS: usize = 1024;
const MAX_PARTS: usize = 65536;
const MAX_RANGES: usize = 4096;
const PARTIAL_LIFETIME: Duration = Duration::from_secs(60);
const DEADLINE: Duration = Duration::from_secs(15);
const MISS: u8 = 0;
const OK: u8 = 1;
const REJECTED: u8 = 2;

type Key = [u8; 32];
type Writer = [u8; 16];

struct Request {
    verb: u8,
    key: Key,
    writer: Writer,
    heads: usize,
    head_bytes: usize,
    part_bytes: usize,
    ranges: Vec<Range<usize>>,
    peers: Option<BTreeMap<IpAddr, u64>>,
}

impl Request {
    async fn read(stream: &mut TcpStream, magic: [u8; 8]) -> io::Result<Self> {
        let invalid = || io::Error::new(io::ErrorKind::InvalidData, "invalid KV request");
        let mut header = [0; HEADER];
        stream.read_exact(&mut header[8..]).await?;
        let ranged = match &magic {
            MAGIC | PEER_READ_MAGIC => false,
            RANGED_MAGIC => true,
            _ => return Err(invalid()),
        };
        let field = |index: usize| {
            let start = 57 + 4 * index;
            u32::from_le_bytes(header[start..start + 4].try_into().unwrap()) as usize
        };
        let heads = field(0);
        let head_bytes = field(3);
        let part_bytes = if ranged { field(2) } else { head_bytes };
        let Some(block_bytes) = heads.checked_mul(head_bytes) else {
            return Err(invalid());
        };
        if !(1..=MAX_HEADS).contains(&heads)
            || head_bytes == 0
            || part_bytes == 0
            || !head_bytes.is_multiple_of(part_bytes)
            || block_bytes > MAX_BLOCK_BYTES
            || block_bytes / part_bytes > MAX_PARTS
        {
            return Err(invalid());
        }
        let range = |first: usize, count: usize| {
            let end = first.checked_add(count)?;
            (count != 0 && end <= block_bytes / part_bytes)
                .then(|| first * part_bytes..end * part_bytes)
        };
        let peers = if &magic == PEER_READ_MAGIC {
            if header[8] != b'h' || !(1..=MAX_PEERS).contains(&field(1)) || field(2) != 0 {
                return Err(invalid());
            }
            Some(read_peers(stream, field(1)).await?)
        } else {
            None
        };
        let ranges = match (header[8], ranged) {
            (b'h', false) if peers.is_some() => Vec::new(),
            (b'H' | b'h', true) if field(1) == 0 => Vec::new(),
            (b'H' | b'h', false) if field(1) == 0 && field(2) == 0 => Vec::new(),
            (b'G' | b'g' | b'P' | b'p', false) => {
                vec![range(field(1), field(2)).ok_or_else(invalid)?]
            }
            (b'G' | b'g' | b'P' | b'p', true) if (1..=MAX_RANGES).contains(&field(1)) => {
                let mut table = vec![0; field(1) * 8];
                stream.read_exact(&mut table).await?;
                let mut end = 0;
                let mut ranges = Vec::with_capacity(field(1));
                for entry in table.chunks_exact(8) {
                    let first = u32::from_le_bytes(entry[..4].try_into().unwrap()) as usize;
                    let count = u32::from_le_bytes(entry[4..].try_into().unwrap()) as usize;
                    let next = range(first, count).ok_or_else(invalid)?;
                    if next.start < end {
                        return Err(invalid());
                    }
                    end = next.end;
                    ranges.push(next);
                }
                ranges
            }
            _ => return Err(invalid()),
        };
        let writer: Writer = header[41..57].try_into().unwrap();
        if matches!(header[8], b'h' | b'g') && writer == [0; 16] {
            return Err(invalid());
        }
        Ok(Self {
            verb: header[8],
            key: header[9..41].try_into().unwrap(),
            writer,
            heads,
            head_bytes,
            part_bytes,
            ranges,
            peers,
        })
    }

    fn payload_bytes(&self) -> usize {
        self.ranges.iter().map(Range::len).sum()
    }

    fn parts(&self) -> usize {
        self.heads * (self.head_bytes / self.part_bytes)
    }

    fn reserved(&self) -> usize {
        self.heads * self.head_bytes + self.parts() * size_of::<bool>()
    }

    fn part_indices(&self) -> impl Iterator<Item = usize> + Clone + '_ {
        self.ranges
            .iter()
            .flat_map(|range| range.start / self.part_bytes..range.end / self.part_bytes)
    }
}

fn unicast_peer(address: IpAddr) -> bool {
    !address.is_unspecified()
        && !address.is_multicast()
        && !matches!(address, IpAddr::V4(ip) if ip.is_broadcast())
}

async fn read_peers(stream: &mut TcpStream, count: usize) -> io::Result<BTreeMap<IpAddr, u64>> {
    let mut table = vec![0; count * PEER_ENTRY];
    stream.read_exact(&mut table).await?;
    let mut peers = BTreeMap::new();
    for entry in table.chunks_exact(PEER_ENTRY) {
        let ip =
            IpAddr::V6(Ipv6Addr::from(<[u8; 16]>::try_from(&entry[..16]).unwrap())).to_canonical();
        let bytes = u64::from_le_bytes(entry[16..].try_into().unwrap());
        if !unicast_peer(ip) || bytes == 0 || peers.insert(ip, bytes).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid KV read peer",
            ));
        }
    }
    Ok(peers)
}

struct Block {
    writer: Writer,
    expires: Instant,
    heads: usize,
    head_bytes: usize,
    part_bytes: usize,
    data: BlockData,
    present: usize,
    reserved: usize,
}

enum BlockData {
    Staging { bytes: Vec<u8>, received: Vec<bool> },
    Published(Bytes),
}

impl Block {
    fn matches(&self, request: &Request) -> bool {
        self.heads == request.heads && self.head_bytes == request.head_bytes
    }

    fn complete(&self) -> bool {
        matches!(self.data, BlockData::Published(_))
    }
}

struct Store {
    blocks: HashMap<Key, Block>,
    capacity: usize,
    used: usize,
}

impl Store {
    fn new(capacity: usize) -> Self {
        Self {
            blocks: HashMap::new(),
            capacity,
            used: 0,
        }
    }

    fn get(&self, request: &Request) -> Option<Vec<Bytes>> {
        let block = self.blocks.get(&request.key)?;
        if !block.matches(request) {
            return None;
        }
        let BlockData::Published(bytes) = &block.data else {
            return None;
        };
        Some(
            request
                .ranges
                .iter()
                .map(|range| bytes.slice(range.clone()))
                .collect(),
        )
    }

    fn put(&mut self, request: &Request, payload: Bytes, now: Instant) -> bool {
        if self
            .blocks
            .get(&request.key)
            .is_some_and(|block| !block.complete() && now >= block.expires)
        {
            self.used -= self
                .blocks
                .remove(&request.key)
                .expect("expired block")
                .reserved;
        }
        if !self.blocks.contains_key(&request.key) {
            let reserved = request.reserved();
            if reserved > self.capacity - self.used {
                self.blocks.retain(|_, block| {
                    let keep = block.complete() || now < block.expires;
                    if !keep {
                        self.used -= block.reserved;
                    }
                    keep
                });
            }
            if reserved > self.capacity - self.used {
                return false;
            }
            self.blocks.insert(
                request.key,
                Block {
                    writer: request.writer,
                    expires: now + PARTIAL_LIFETIME,
                    heads: request.heads,
                    head_bytes: request.head_bytes,
                    part_bytes: request.part_bytes,
                    data: BlockData::Staging {
                        bytes: vec![0; request.heads * request.head_bytes],
                        received: vec![false; request.parts()],
                    },
                    present: 0,
                    reserved,
                },
            );
            self.used += reserved;
        }

        let block = self.blocks.get_mut(&request.key).expect("reserved block");
        if !block.matches(request) {
            return false;
        }
        if let BlockData::Published(bytes) = &block.data {
            let mut remaining = payload.as_ref();
            for range in &request.ranges {
                let Some(rest) = remaining.strip_prefix(&bytes[range.clone()]) else {
                    return false;
                };
                remaining = rest;
            }
            return remaining.is_empty();
        }
        if block.writer != request.writer || block.part_bytes != request.part_bytes {
            return false;
        }
        let BlockData::Staging { bytes, received } = &mut block.data else {
            unreachable!("published blocks were checked already");
        };
        let target = |index| {
            let start = index * request.part_bytes;
            start..start + request.part_bytes
        };
        let parts = request.part_indices();
        if parts
            .clone()
            .zip(payload.chunks_exact(request.part_bytes))
            .any(|(index, incoming)| received[index] && bytes[target(index)] != *incoming)
        {
            return false;
        }
        for (index, incoming) in parts.zip(payload.chunks_exact(request.part_bytes)) {
            if !received[index] {
                bytes[target(index)].copy_from_slice(incoming);
                received[index] = true;
                block.present += 1;
            }
        }
        if block.present == received.len() {
            block.data = BlockData::Published(Bytes::from(std::mem::take(bytes)));
        }
        true
    }
}

async fn exchange(
    mut stream: TcpStream,
    peer: IpAddr,
    store: &Mutex<Store>,
    capacity: usize,
    checkpoints: &Checkpoints,
) -> io::Result<()> {
    let mut magic = [0; 8];
    stream.read_exact(&mut magic).await?;
    if &magic == METRICS_MAGIC {
        let body = checkpoints.snapshot(Instant::now());
        stream.write_u32(body.len() as u32).await?;
        return stream.write_all(&body).await;
    }
    let request = Request::read(&mut stream, magic).await?;
    let bytes = request.payload_bytes();
    if matches!(request.verb, b'P' | b'p') {
        if request.reserved() > capacity {
            return stream.write_u8(REJECTED).await;
        }
        let transfer = if request.verb == b'p' {
            let Some(transfer) = checkpoints.write(peer, bytes as u64, Instant::now()) else {
                return stream.write_u8(REJECTED).await;
            };
            stream.write_u8(OK).await?;
            Some(transfer)
        } else {
            None
        };
        let mut payload = vec![0; bytes];
        stream.read_exact(&mut payload).await?;
        if let Some(transfer) = transfer {
            transfer.complete();
        }
        let stored =
            store
                .lock()
                .expect("KV store lock")
                .put(&request, payload.into(), Instant::now());
        return stream.write_u8(if stored { OK } else { REJECTED }).await;
    }
    let heads = store.lock().expect("KV store lock").get(&request);
    let Some(heads) = heads else {
        return stream.write_u8(MISS).await;
    };
    if request.verb == b'h' {
        let reserved = checkpoints.reserve_read(
            request.key,
            request.writer,
            (request.heads * request.head_bytes) as u64,
            request.peers,
            Instant::now(),
        );
        return stream.write_u8(if reserved { OK } else { REJECTED }).await;
    }
    let transfer = if request.verb == b'g' {
        let Some(transfer) = checkpoints.read(
            request.key,
            request.writer,
            peer,
            bytes as u64,
            Instant::now(),
        ) else {
            return stream.write_u8(REJECTED).await;
        };
        Some(transfer)
    } else {
        None
    };
    stream.write_u8(OK).await?;
    for head in heads {
        stream.write_all(&head).await?;
    }
    if let Some(transfer) = transfer {
        transfer.complete();
    }
    Ok(())
}

#[tokio::main]
async fn main() {
    if let Err(error) = run(std::env::args().skip(1)).await {
        eprintln!("vllm-plait-kv-store: {error}");
        std::process::exit(1);
    }
}

async fn run(mut args: impl Iterator<Item = String>) -> Result<(), Box<dyn std::error::Error>> {
    let first = args.next();
    if first.as_deref() == Some("--stats") {
        let addr = args.next().ok_or("--stats needs IP:port")?.parse()?;
        if args.next().is_some() {
            return Err("--stats accepts a single store address".into());
        }
        let body = tokio::time::timeout(DEADLINE, read_stats(addr)).await??;
        println!("{body}");
        return Ok(());
    }
    if first.as_deref() == Some("--audit-checkpoint-peer-budget-plan") {
        let path = args
            .next()
            .ok_or("--audit-checkpoint-peer-budget-plan needs a file")?;
        if args.next().is_some() {
            return Err("--audit-checkpoint-peer-budget-plan accepts a single file".into());
        }
        let report = PeerPlan::load(&path)?.audit().await?;
        println!("{report}");
        return Ok(());
    }
    if let Some(
        command @ ("--prepare-checkpoint-peer-budget-plan"
        | "--activate-checkpoint-peer-budget-plan"),
    ) = first.as_deref()
    {
        #[cfg(unix)]
        return budget_control::cli(command == "--activate-checkpoint-peer-budget-plan", args)
            .await;
        #[cfg(not(unix))]
        return Err(format!("{command} requires a Unix control socket").into());
    }
    let mut args = first.into_iter().chain(args);
    let mut listen = "127.0.0.1:7952".parse::<SocketAddr>()?;
    let mut capacity = 256 << 20;
    let mut checkpoint_rate = None;
    let mut checkpoint_burst = None;
    let mut peers = BTreeMap::new();
    let mut plan_path = None;
    let mut control_dir = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--listen" => listen = args.next().ok_or("--listen needs IP:port")?.parse()?,
            "--capacity-bytes" => {
                capacity = args
                    .next()
                    .ok_or("--capacity-bytes needs a size")?
                    .parse::<usize>()?;
            }
            "--checkpoint-bytes-per-second" => {
                checkpoint_rate = Some(
                    args.next()
                        .ok_or("--checkpoint-bytes-per-second needs a rate")?
                        .parse::<u64>()?,
                );
            }
            "--checkpoint-burst-bytes" => {
                checkpoint_burst = Some(
                    args.next()
                        .ok_or("--checkpoint-burst-bytes needs a size")?
                        .parse::<u64>()?,
                );
            }
            "--checkpoint-peer-limits" => {
                if !peers.is_empty() {
                    return Err("the checkpoint peer map may only be supplied once".into());
                }
                peers = read_peer_limits(
                    &args
                        .next()
                        .ok_or("--checkpoint-peer-limits needs a JSON file")?,
                )?;
            }
            "--checkpoint-peer-budget-plan" => {
                if plan_path.is_some() {
                    return Err("the checkpoint budget plan may only be supplied once".into());
                }
                plan_path = Some(
                    args.next()
                        .ok_or("--checkpoint-peer-budget-plan needs a file")?,
                );
            }
            "--checkpoint-budget-control" => {
                if control_dir.is_some() {
                    return Err(
                        "the checkpoint budget control directory may only be supplied once".into(),
                    );
                }
                control_dir = Some(
                    args.next()
                        .ok_or("--checkpoint-budget-control needs a directory")?,
                );
            }
            _ => return Err(format!("unknown KV store argument {arg}").into()),
        }
    }
    if capacity == 0 {
        return Err("KV store capacity must be positive".into());
    }
    let limits = match (checkpoint_rate, checkpoint_burst) {
        (None, None) => None,
        (Some(bytes_per_second), Some(burst_bytes)) if bytes_per_second > 0 && burst_bytes > 0 => {
            Some(Limit {
                bytes_per_second,
                burst_bytes,
            })
        }
        _ => return Err("checkpoint byte rate and burst must both be positive".into()),
    };
    let listener = TcpListener::bind(listen).await?;
    let checkpoints = if let Some(path) = plan_path {
        if !peers.is_empty() {
            return Err("choose a checkpoint peer map or a shared budget plan".into());
        }
        if let Some(dir) = control_dir {
            #[cfg(unix)]
            {
                budget_control::Control::open(dir.into(), listen, &path)?.start(limits)?
            }
            #[cfg(not(unix))]
            return Err(format!("checkpoint budget control directory {dir} requires Unix").into());
        } else {
            let (membership, peers) = PeerPlan::load(&path)?.share_for(listen)?;
            Arc::new(Checkpoints::new(
                limits,
                peers,
                Instant::now(),
                Some(membership),
            ))
        }
    } else {
        if control_dir.is_some() {
            return Err(
                "--checkpoint-budget-control requires --checkpoint-peer-budget-plan".into(),
            );
        }
        Arc::new(Checkpoints::new(limits, peers, Instant::now(), None))
    };
    let store = Arc::new(Mutex::new(Store::new(capacity)));
    let connections = Arc::new(Semaphore::new(32));
    println!("Plait KV store listening on {}", listener.local_addr()?);
    loop {
        let permit = connections.clone().acquire_owned().await?;
        let (stream, remote) = listener.accept().await?;
        stream.set_nodelay(true)?;
        let store = store.clone();
        let checkpoints = checkpoints.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let _ = tokio::time::timeout(
                DEADLINE,
                exchange(
                    stream,
                    remote.ip().to_canonical(),
                    &store,
                    capacity,
                    &checkpoints,
                ),
            )
            .await;
        });
    }
}

fn read_peer_limits(path: &str) -> Result<BTreeMap<IpAddr, PeerLimit>, Box<dyn std::error::Error>> {
    let input: BTreeMap<IpAddr, PeerLimit> = serde_json::from_reader(std::fs::File::open(path)?)?;
    if !(1..=MAX_PEERS).contains(&input.len()) {
        return Err("invalid number of checkpoint peers".into());
    }
    let mut peers = BTreeMap::new();
    for (address, limits) in input {
        let address = address.to_canonical();
        if !unicast_peer(address)
            || !limits.read.valid()
            || !limits.write.valid()
            || peers.insert(address, limits).is_some()
        {
            return Err("invalid or duplicate checkpoint peer limit".into());
        }
    }
    Ok(peers)
}

async fn read_stats(addr: SocketAddr) -> io::Result<String> {
    let mut stream = TcpStream::connect(addr).await?;
    stream.write_all(METRICS_MAGIC).await?;
    let size = stream.read_u32().await? as usize;
    if !(1..=MAX_METRICS_BYTES).contains(&size) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid KV store metrics size",
        ));
    }
    let mut body = vec![0; size];
    stream.read_exact(&mut body).await?;
    String::from_utf8(body).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn open_request(addr: SocketAddr, verb: u8, first: u32, count: u32) -> TcpStream {
        let mut header = Vec::from(*MAGIC);
        header.push(verb);
        header.extend_from_slice(&[3; 32]);
        header.extend_from_slice(&[7; 16]);
        for field in [2, first, count, 2] {
            header.extend_from_slice(&field.to_le_bytes());
        }
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(&header).await.unwrap();
        stream
    }

    #[tokio::test]
    async fn refusals_precede_optional_payload_and_leave_mandatory_handoff_available() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let store = Mutex::new(Store::new(64));
            let budget = Checkpoints::new(
                Some(Limit {
                    bytes_per_second: 1,
                    burst_bytes: 4,
                }),
                BTreeMap::new(),
                Instant::now(),
                None,
            );
            for _ in 0..7 {
                let (stream, peer) = listener.accept().await.unwrap();
                tokio::time::timeout(
                    Duration::from_secs(2),
                    exchange(stream, peer.ip(), &store, 64, &budget),
                )
                .await
                .unwrap()
                .unwrap();
            }
        });

        let mut write = open_request(addr, b'P', 0, 2).await;
        write.write_all(&[1, 2, 3, 4]).await.unwrap();
        assert_eq!(write.read_u8().await.unwrap(), OK);
        assert_eq!(
            open_request(addr, b'h', 0, 0)
                .await
                .read_u8()
                .await
                .unwrap(),
            OK
        );
        assert_eq!(
            open_request(addr, b'p', 0, 1)
                .await
                .read_u8()
                .await
                .unwrap(),
            REJECTED
        );
        for (verb, first, count, expected) in [
            (b'G', 0, 2, &[1, 2, 3, 4][..]),
            (b'g', 0, 1, &[1, 2][..]),
            (b'g', 1, 1, &[3, 4][..]),
        ] {
            let mut reply = open_request(addr, verb, first, count).await;
            assert_eq!(reply.read_u8().await.unwrap(), OK);
            let mut payload = Vec::new();
            reply.read_to_end(&mut payload).await.unwrap();
            assert_eq!(payload, expected);
        }
        let stats = read_stats(addr).await.unwrap();
        let stats: serde_json::Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(stats["read_bytes"], 4);
        assert_eq!(stats["write_bytes"], 0);
        assert_eq!(stats["write_reservations_denied"], 1);
        assert_eq!(stats["pending_bytes"], 0);
        server.await.unwrap();
    }

    fn request(first: usize, count: usize, writer: u8) -> Request {
        Request {
            verb: b'P',
            key: [1; 32],
            writer: [writer; 16],
            heads: 4,
            head_bytes: 1,
            part_bytes: 1,
            ranges: vec![first..first + count],
            peers: None,
        }
    }

    #[test]
    fn exposes_only_complete_blocks_and_reads_in_logical_head_order() {
        let mut store = Store::new(1024);
        let now = Instant::now();
        assert!(store.put(&request(2, 2, 1), Bytes::from_static(&[30, 40]), now));
        assert!(store.get(&request(2, 1, 1)).is_none());
        assert!(!store.put(&request(0, 2, 2), Bytes::from_static(&[10, 20]), now));
        assert!(store.put(&request(0, 2, 1), Bytes::from_static(&[10, 20]), now));
        let heads = store.get(&request(1, 3, 2)).unwrap();
        assert_eq!(heads.concat(), [20, 30, 40]);
        assert!(!store.put(&request(2, 1, 2), Bytes::from_static(&[90]), now));
    }

    #[test]
    fn an_incomplete_write_expires_without_leaking_its_heads_into_its_successor() {
        let mut store = Store::new(request(0, 4, 1).reserved());
        let now = Instant::now();
        assert!(store.put(&request(0, 2, 1), Bytes::from_static(&[1, 2]), now));
        let later = now + PARTIAL_LIFETIME;
        assert!(store.put(&request(2, 2, 2), Bytes::from_static(&[7, 8]), later));
        assert!(store.get(&request(2, 1, 2)).is_none());
        assert!(store.put(&request(0, 2, 2), Bytes::from_static(&[5, 6]), later));
        let heads = store.get(&request(0, 4, 2)).unwrap();
        assert_eq!(heads.concat(), [5, 6, 7, 8]);
    }

    #[test]
    fn pipeline_ranges_share_one_writer_and_retain_legacy_head_order() {
        let mut store = Store::new(1024);
        let now = Instant::now();
        let ranges = |spans, part_bytes, writer| Request {
            verb: b'P',
            key: [2; 32],
            writer: [writer; 16],
            heads: 2,
            head_bytes: 4,
            part_bytes,
            ranges: spans,
            peers: None,
        };
        let first_stage = ranges(vec![0..2, 4..6], 2, 1);
        let next_stage = ranges(vec![2..4, 6..8], 2, 1);
        let old_head = ranges(vec![4..8], 4, 1);
        let other_partition = ranges(vec![0..4, 6..8], 2, 1);
        assert!(store.put(&next_stage, Bytes::from_static(&[3, 4, 7, 8]), now));
        assert!(store.get(&first_stage).is_none());
        assert!(store.get(&old_head).is_none());
        assert!(!store.put(&old_head, Bytes::from_static(&[5, 6, 7, 8]), now));
        let rival = ranges(vec![0..2, 4..6], 2, 2);
        assert!(!store.put(&rival, Bytes::from_static(&[1, 2, 5, 6]), now));
        assert!(store.put(&first_stage, Bytes::from_static(&[1, 2, 5, 6]), now));
        assert_eq!(store.get(&old_head).unwrap().concat(), [5, 6, 7, 8]);
        assert_eq!(
            store.get(&other_partition).unwrap().concat(),
            [1, 2, 3, 4, 7, 8]
        );
        assert!(store.put(&old_head, Bytes::from_static(&[5, 6, 7, 8]), now));
        assert!(!store.put(&old_head, Bytes::from_static(&[9, 6, 7, 8]), now));

        let mut store = Store::new(1024);
        let all_heads = ranges(vec![0..8], 4, 3);
        assert!(store.put(
            &all_heads,
            Bytes::from_static(&[1, 2, 3, 4, 5, 6, 7, 8]),
            now
        ));
        assert_eq!(store.get(&next_stage).unwrap().concat(), [3, 4, 7, 8]);
    }
}
