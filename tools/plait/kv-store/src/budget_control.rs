//! Durable, online replacement of checkpoint allocations on fixed KV stores.

use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::SocketAddr;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::time::timeout;

use super::DEADLINE;
use super::checkpoints::{Checkpoints, Limit};
use super::peer_plan::{BudgetEpoch, Document, MAX_PLAN_BYTES, PeerPlan, canonical_endpoint};

const STATE_FILE: &str = "budget-v1.json";
const SOCKET: &str = "control.sock";
const MAX_FRAME: u64 = 2 * MAX_PLAN_BYTES + (64 << 10);
const COMMAND_DEADLINE: Duration = Duration::from_secs(35);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record<T> {
    endpoint: SocketAddr,
    generation: u64,
    previous_plan_id: Option<String>,
    active: T,
    target: Option<T>,
}

impl Record<PeerPlan> {
    fn epoch(&self) -> BudgetEpoch {
        BudgetEpoch {
            generation: self.generation,
            previous_plan_id: self.previous_plan_id.clone(),
            target_plan_id: self.target.as_ref().map(|plan| plan.id().to_owned()),
        }
    }

    fn documents(&self) -> Record<&Document> {
        Record {
            endpoint: self.endpoint,
            generation: self.generation,
            previous_plan_id: self.previous_plan_id.clone(),
            active: self.active.document(),
            target: self.target.as_ref().map(PeerPlan::document),
        }
    }
}

impl Record<Document> {
    fn validate(self) -> io::Result<Record<PeerPlan>> {
        let active = PeerPlan::validate(self.active)?;
        let target = self.target.map(PeerPlan::validate).transpose()?;
        if let Some(target) = &target {
            active.validate_successor(target)?;
        }
        let valid_lineage = match &self.previous_plan_id {
            None => self.generation == 0,
            Some(id) => {
                self.generation > 0
                    && id != active.id()
                    && id.len() == 64
                    && id.bytes().all(|byte| byte.is_ascii_hexdigit())
            }
        };
        if !valid_lineage {
            return Err(invalid("invalid persisted checkpoint budget lineage"));
        }
        Ok(Record {
            endpoint: self.endpoint,
            generation: self.generation,
            previous_plan_id: self.previous_plan_id,
            active,
            target,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct Command<T> {
    generation: u64,
    old_plan_id: String,
    change: Change<T>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Change<T> {
    Prepare(T),
    Activate { target_plan_id: String },
}

#[derive(Serialize, Deserialize)]
enum Reply {
    Epoch(BudgetEpoch),
    Rejected(String),
}

pub(super) struct Control {
    directory: PathBuf,
    listener: UnixListener,
    state: Record<PeerPlan>,
    poisoned: bool,
    _lock: File,
}

impl Control {
    pub(super) fn open(
        directory: PathBuf,
        endpoint: SocketAddr,
        bootstrap: &str,
    ) -> io::Result<Self> {
        let endpoint = canonical_endpoint(endpoint);
        private_directory(&directory)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(directory.join("writer.lock"))?;
        // SAFETY: the descriptor is valid and the controller owns the lock file.
        if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let state = match File::open(directory.join(STATE_FILE)) {
            Ok(file) => read_record(file)?.validate()?,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                let state = Record {
                    endpoint,
                    generation: 0,
                    previous_plan_id: None,
                    active: PeerPlan::load(bootstrap)?,
                    target: None,
                };
                state.active.share_for(endpoint)?;
                persist(&directory, &serde_json::to_vec(&state.documents())?)?;
                state
            }
            Err(err) => return Err(err),
        };
        if state.endpoint != endpoint {
            return Err(invalid(
                "checkpoint budget control directory belongs to a different store",
            ));
        }
        state.active.share_for(endpoint)?;
        // A process may have exited after rename and before syncing the directory.
        File::open(&directory)?.sync_all()?;
        let socket = directory.join(SOCKET);
        match fs::symlink_metadata(&socket) {
            Ok(metadata) if metadata.file_type().is_socket() => fs::remove_file(&socket)?,
            Ok(_) => return Err(invalid("checkpoint budget control path is not a socket")),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {}
            Err(err) => return Err(err),
        }
        let listener = UnixListener::bind(&socket)?;
        fs::set_permissions(socket, fs::Permissions::from_mode(0o600))?;
        Ok(Self {
            directory,
            listener,
            state,
            poisoned: false,
            _lock: lock,
        })
    }

    pub(super) fn start(self, limit: Option<Limit>) -> io::Result<Arc<Checkpoints>> {
        let (membership, peers) = self.state.active.share_for(self.state.endpoint)?;
        let checkpoints = Arc::new(Checkpoints::new(
            limit,
            peers,
            Instant::now(),
            Some(membership),
        ));
        checkpoints.set_epoch(self.state.epoch());
        tokio::spawn(self.serve(checkpoints.clone()));
        Ok(checkpoints)
    }

    async fn serve(mut self, checkpoints: Arc<Checkpoints>) {
        loop {
            let mut stream = match self.listener.accept().await {
                Ok((stream, _)) => stream,
                Err(err) => {
                    eprintln!("checkpoint budget control accept: {err}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
            };
            // One command at a time. During activation, this store must keep
            // advertising the same target throughout its peer observations.
            let response = match timeout(DEADLINE, receive(&mut stream)).await {
                Ok(Ok(command)) => match self.apply(command, &checkpoints).await {
                    Ok(epoch) => Reply::Epoch(epoch),
                    Err(err) => Reply::Rejected(err.to_string()),
                },
                Ok(Err(err)) => Reply::Rejected(err.to_string()),
                Err(err) => Reply::Rejected(err.to_string()),
            };
            let _ = timeout(DEADLINE, send(&mut stream, &response)).await;
        }
    }

    async fn apply(
        &mut self,
        command: Command<Document>,
        checkpoints: &Checkpoints,
    ) -> io::Result<BudgetEpoch> {
        if self.poisoned {
            return Err(io::Error::other(
                "checkpoint budget persistence failed; restart this store",
            ));
        }
        let Command {
            generation,
            old_plan_id,
            change,
        } = command;
        let next = generation
            .checked_add(1)
            .ok_or_else(|| invalid("checkpoint budget generation exhausted"))?;
        let (target_plan_id, proposed) = match change {
            Change::Prepare(document) => {
                let target = PeerPlan::validate(document)?;
                (target.id().to_owned(), Some(target))
            }
            Change::Activate { target_plan_id } => (target_plan_id, None),
        };
        if self.state.generation == next
            && self.state.previous_plan_id.as_deref() == Some(old_plan_id.as_str())
            && self.state.active.id() == target_plan_id
        {
            return Ok(self.state.epoch());
        }
        if self.state.generation != generation || self.state.active.id() != old_plan_id {
            return Err(invalid(
                "checkpoint budget predecessor does not match the active generation",
            ));
        }
        if let Some(target) = proposed {
            self.state.active.validate_successor(&target)?;
            if self
                .state
                .target
                .as_ref()
                .is_some_and(|plan| plan.id() == target_plan_id)
            {
                return Ok(self.state.epoch());
            }
            checkpoints.close_admission();
            let saved = Record {
                target: Some(target.document()),
                ..self.state.documents()
            };
            let result = persist_online(&self.directory, &self._lock, &saved).await;
            self.check_persistence(checkpoints, result)?;
            self.state.target = Some(target);
            let epoch = self.state.epoch();
            checkpoints.set_epoch(epoch.clone());
            return Ok(epoch);
        }

        let target = self
            .state
            .target
            .as_ref()
            .filter(|plan| plan.id() == target_plan_id)
            .ok_or_else(|| invalid("checkpoint budget successor has not been prepared"))?;
        if !checkpoints.drained(Instant::now()) {
            return Err(io::Error::other(
                "local checkpoint reservations are still draining",
            ));
        }
        let endpoint = self.state.endpoint;
        self.state
            .active
            .require_drained(endpoint, generation, target)
            .await?;
        let (membership, peers) = target.share_for(endpoint)?;
        let saved = Record {
            endpoint,
            generation: next,
            previous_plan_id: Some(old_plan_id),
            active: target.document(),
            target: None,
        };
        let result = persist_online(&self.directory, &self._lock, &saved).await;
        let previous_plan_id = saved.previous_plan_id;
        self.check_persistence(checkpoints, result)?;
        self.state = Record {
            endpoint,
            generation: next,
            previous_plan_id,
            active: self.state.target.take().expect("prepared successor"),
            target: None,
        };
        let epoch = self.state.epoch();
        checkpoints.install_plan(membership, peers, epoch.clone(), Instant::now());
        Ok(epoch)
    }

    fn check_persistence(
        &mut self,
        checkpoints: &Checkpoints,
        result: io::Result<()>,
    ) -> io::Result<()> {
        if result.is_err() {
            self.poisoned = true;
            checkpoints.mark_epoch_uncertain();
        }
        result
    }
}

fn private_directory(path: &Path) -> io::Result<()> {
    match DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => {
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty());
            File::open(parent.unwrap_or_else(|| Path::new(".")))?.sync_all()?;
        }
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => {}
        Err(err) => return Err(err),
    }
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "checkpoint budget control requires a private directory",
        ));
    }
    Ok(())
}

async fn persist_online(
    directory: &Path,
    lock: &File,
    state: &Record<&Document>,
) -> io::Result<()> {
    let data = serde_json::to_vec(state)?;
    let directory = directory.to_owned();
    let lock = lock.try_clone()?;
    tokio::task::spawn_blocking(move || {
        let _lock = lock;
        persist(&directory, &data)
    })
    .await
    .map_err(io::Error::other)?
}

fn persist(directory: &Path, data: &[u8]) -> io::Result<()> {
    let temp = directory.join("budget.next");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&temp)?;
    file.write_all(data)?;
    file.sync_all()?;
    fs::rename(temp, directory.join(STATE_FILE))?;
    File::open(directory)?.sync_all()
}

fn read_record(file: File) -> io::Result<Record<Document>> {
    let mut payload = Vec::new();
    file.take(MAX_FRAME + 1).read_to_end(&mut payload)?;
    if payload.len() as u64 > MAX_FRAME {
        return Err(invalid("checkpoint budget state is too large"));
    }
    Ok(serde_json::from_slice(&payload)?)
}

async fn receive<T: DeserializeOwned>(stream: &mut UnixStream) -> io::Result<T> {
    let size = stream.read_u32().await?;
    if size == 0 || u64::from(size) > MAX_FRAME {
        return Err(invalid("invalid checkpoint budget command size"));
    }
    let mut payload = vec![0; size as usize];
    stream.read_exact(&mut payload).await?;
    Ok(serde_json::from_slice(&payload)?)
}

async fn send(stream: &mut UnixStream, value: &impl Serialize) -> io::Result<()> {
    let payload = serde_json::to_vec(value)?;
    if payload.len() as u64 > MAX_FRAME {
        return Err(invalid("checkpoint budget command is too large"));
    }
    stream.write_u32(payload.len() as u32).await?;
    stream.write_all(&payload).await
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

pub(super) async fn cli(
    activate: bool,
    mut args: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let directory = PathBuf::from(
        args.next()
            .ok_or("expected checkpoint budget control directory")?,
    );
    let generation = args
        .next()
        .ok_or("expected checkpoint budget generation")?
        .parse()?;
    let old = PeerPlan::load(&args.next().ok_or("expected predecessor budget plan file")?)?;
    let target = PeerPlan::load(&args.next().ok_or("expected successor budget plan file")?)?;
    if args.next().is_some() {
        return Err("expected DIR GENERATION OLD_PLAN NEW_PLAN".into());
    }
    old.validate_successor(&target)?;
    let command = Command {
        generation,
        old_plan_id: old.id().to_owned(),
        change: if activate {
            Change::Activate {
                target_plan_id: target.id().to_owned(),
            }
        } else {
            Change::Prepare(target.document())
        },
    };
    let response = timeout(COMMAND_DEADLINE, async {
        let mut stream = UnixStream::connect(directory.join(SOCKET)).await?;
        send(&mut stream, &command).await?;
        receive(&mut stream).await
    })
    .await??;
    match response {
        Reply::Epoch(epoch) => println!("{}", serde_json::to_string(&epoch)?),
        Reply::Rejected(err) => return Err(err.into()),
    }
    Ok(())
}
