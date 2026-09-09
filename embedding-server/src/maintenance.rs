//! The housekeeping a server owes its shards: fsync, snapshot, compaction and the memory budget.
//!
//! Nothing in the request path does any of this. A shard appends to its log and updates its
//! indexes; putting the log on the device and writing the derived files out is a timer's job, so
//! that a restart replays seconds of log rather than everything since the process started.
//!
//! The pass is a function rather than only a thread so a test can drive it directly. [`spawn`]
//! wraps it in the loop `main` runs.

use embedding_common::config::StorageConfig;
use embedding_store::prelude::ShardPool;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// How often the server does each kind of housekeeping, from `[storage]`.
#[derive(Debug, Clone, Copy)]
pub struct MaintenancePolicy {
    /// `None` means every write is synced as it happens, so the timer has nothing to sync.
    pub fsync_interval: Option<Duration>,
    pub snapshot_interval: Duration,
    pub compact_ratio: f64,
}

impl MaintenancePolicy {
    pub fn from_config(storage: &StorageConfig) -> Self {
        MaintenancePolicy {
            fsync_interval: storage.fsync_interval(),
            snapshot_interval: storage.snapshot_interval(),
            compact_ratio: storage.compact_ratio,
        }
    }

    /// How long the loop waits between passes. Snapshots are far rarer than fsyncs, so the tick is
    /// the fsync interval and the snapshot is done on the passes it falls due.
    pub fn tick(&self) -> Duration {
        self.fsync_interval
            .unwrap_or(self.snapshot_interval)
            .min(self.snapshot_interval)
            .max(Duration::from_millis(50))
    }
}

/// One pass. Every failure is logged and the pass continues: housekeeping that gives up on the
/// first bad shard would leave every other shard unsynced, and the log is still the source of
/// truth whatever happens here.
pub fn maintenance_pass(pool: &ShardPool, policy: &MaintenancePolicy, snapshot_due: bool) {
    if policy.fsync_interval.is_some() {
        if let Err(e) = pool.fsync_all() {
            warn!("fsync pass failed: {:#}", e);
        }
    }

    if !snapshot_due {
        return;
    }

    if let Err(e) = pool.snapshot_all() {
        warn!("snapshot pass failed: {:#}", e);
    }
    // After the snapshot, so a shard that compacts takes its fresh snapshot from a log that was
    // just made durable rather than one that was not.
    if let Err(e) = pool.compact_if_needed(policy.compact_ratio) {
        warn!("compaction pass failed: {:#}", e);
    }
    // Last: everything above may have grown a shard, and eviction wants a shard that is already
    // snapshotted, which is exactly what it now is.
    match pool.enforce_budget() {
        Ok(0) => {}
        Ok(n) => info!("evicted {} shard(s) to get back under budget", n),
        Err(e) => warn!("budget enforcement failed: {:#}", e),
    }
}

/// The timer thread. It never returns; the process exits from the signal handler or the proxy.
pub fn spawn(pool: Arc<ShardPool>, policy: MaintenancePolicy) -> JoinHandle<()> {
    thread::spawn(move || {
        let tick = policy.tick();
        info!(
            "storage maintenance: tick {:?}, fsync {:?}, snapshot every {:?}, compact above {}",
            tick, policy.fsync_interval, policy.snapshot_interval, policy.compact_ratio
        );
        let mut last_snapshot = Instant::now();
        loop {
            thread::sleep(tick);
            let snapshot_due = last_snapshot.elapsed() >= policy.snapshot_interval;
            maintenance_pass(&pool, &policy, snapshot_due);
            if snapshot_due {
                last_snapshot = Instant::now();
            }
        }
    })
}

/// Snapshot every loaded shard and exit. Installed for SIGINT, SIGTERM and SIGHUP.
///
/// Without it a restart replays everything written since the last timer snapshot. That is correct
/// — the log is the source of truth — but it is the difference between a restart that loads its
/// indexes and one that rebuilds them.
pub fn install_shutdown_handler(pool: Arc<ShardPool>) -> anyhow::Result<()> {
    ctrlc::set_handler(move || {
        // Claimed before anything else, so the main thread can tell a signal apart from a real
        // proxy failure and wait rather than exiting out from under this snapshot.
        SHUTTING_DOWN.store(true, Ordering::SeqCst);
        info!("shutting down: snapshotting every loaded shard");
        match pool.snapshot_all() {
            Ok(n) => info!("snapshotted {} shard(s), exiting", n),
            // Exit anyway: the log holds every acknowledged write, so the cost of a failed
            // snapshot is a slow restart, not a lost document.
            Err(e) => warn!("snapshot on shutdown failed, exiting anyway: {:#}", e),
        }
        // The ZMQ proxy owns the main thread and never returns, so there is nothing to unwind to.
        std::process::exit(0);
    })?;
    Ok(())
}

/// Set by the shutdown handler before it starts snapshotting.
static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);

/// Whether a shutdown signal has been taken.
pub fn is_shutting_down() -> bool {
    SHUTTING_DOWN.load(Ordering::SeqCst)
}

/// How long the main thread waits for the handler to claim a shutdown. The handler sets the flag
/// before it does anything else, so this only has to cover the scheduling gap between the signal
/// interrupting the proxy and the handler thread being run.
const SHUTDOWN_CLAIM_TIMEOUT: Duration = Duration::from_secs(2);

/// How long it then waits for the snapshot to finish. A large shard's usearch save is measured in
/// seconds, so this is deliberately generous: it is not a deadline for the snapshot, only a stop
/// so that a signal which never reaches the handler cannot hang the server forever.
const SHUTDOWN_EXIT_TIMEOUT: Duration = Duration::from_secs(300);

/// Wait out a shutdown that has interrupted the ZMQ proxy, and say whether there was one.
///
/// A signal interrupts [`zmq::proxy`] on the main thread at the same moment the handler starts
/// snapshotting on its own, and whichever finishes first decides the process's fate. Returning
/// from `main` there ended the process mid-snapshot and exited non-zero on an ordinary SIGTERM —
/// the log still held every acknowledged write, but the next start had to replay it, which is the
/// whole thing the shutdown snapshot exists to avoid.
///
/// So the main thread waits here instead. The handler's own `exit(0)` is what normally ends the
/// process; this returning at all means the handler is stuck, and `true` says to go quietly
/// anyway because a shutdown was asked for. `false` means the interruption was not a shutdown and
/// the caller should treat it as the error it is.
pub fn await_shutdown() -> bool {
    let claim = Instant::now();
    while !is_shutting_down() {
        if claim.elapsed() >= SHUTDOWN_CLAIM_TIMEOUT {
            return false;
        }
        thread::sleep(Duration::from_millis(10));
    }

    let start = Instant::now();
    while start.elapsed() < SHUTDOWN_EXIT_TIMEOUT {
        thread::sleep(Duration::from_millis(50));
    }
    warn!(
        "shutdown handler has not finished after {:?}; exiting without waiting further",
        SHUTDOWN_EXIT_TIMEOUT
    );
    true
}
