//! `ShardPool`: one [`Shard`] per workspace, under one memory budget.
//!
//! A workspace's shard lives at `<dir>/<uuid>/`. [`ShardPool::get`] opens it on first use (or
//! creates it empty) and hands back an `Arc`; when the loaded shards' index residency exceeds the
//! budget, the least recently used ones that no caller is holding are snapshotted and dropped.
//! The next `get` reopens them from that snapshot, which is why eviction costs a reload rather
//! than a rebuild.
//!
//! What the budget does and does not cover:
//!
//! - It is enforced across shards. A single shard larger than the whole budget stays loaded,
//!   because there is nothing smaller to evict.
//! - It counts what the two usearch indexes hold resident, which dominates (about 1.7 KB per
//!   vector at 768 dimensions and f16). The record maps and the mapped segments are not counted;
//!   a mapped segment is the page cache's to reclaim, not ours.
//! - Shards a caller still holds an `Arc` to are never evicted. Drop the `Arc` when the request
//!   is done, or hold it deliberately to pin a shard.
//!
//! Two things the pool is careful about, both because opening and snapshotting a shard are slow
//! enough to be measured in seconds:
//!
//! - **No shard is opened with the pool lock held.** One workspace rebuilding its indexes from a
//!   500k-record log would otherwise stall every other workspace's requests behind it.
//! - **A live shard never leaves the map.** Eviction snapshots through a clone and only then
//!   removes the entry, so no `get` can open a second `Shard` on a directory whose first one is
//!   still alive. Two `Shard`s on one log would each own the active segment.

use crate::shard::{Shard, ShardOptions};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use tracing::{debug, info, warn};
use uuid::Uuid;

pub struct ShardPool {
    dir: PathBuf,
    opts: ShardOptions,
    budget_bytes: usize,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    shards: HashMap<Uuid, Entry>,
    /// One gate per workspace with an open in flight, so a second caller for the same workspace
    /// waits instead of opening the same directory a second time. An entry lives only as long as
    /// some thread holds it: [`ShardPool::retire_gates`] drops the rest.
    opening: HashMap<Uuid, Arc<Mutex<()>>>,
    /// Logical clock for least-recently-used ordering.
    tick: u64,
}

struct Entry {
    shard: Arc<Shard>,
    last_used: u64,
    /// Index residency as of the last measurement that did not have to wait for the shard's own
    /// lock. Stale by at most one budget pass, which is what the budget is accurate to anyway.
    bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Shards currently in memory.
    pub loaded: usize,
    /// What their indexes hold resident.
    pub memory_bytes: usize,
    pub budget_bytes: usize,
}

impl ShardPool {
    /// A pool over `dir`, with no memory budget. Nothing is opened until [`get`](Self::get).
    pub fn new(dir: impl Into<PathBuf>, opts: ShardOptions) -> Result<Self> {
        let dir = dir.into();
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(ShardPool {
            dir,
            opts,
            budget_bytes: usize::MAX,
            inner: Mutex::new(Inner::default()),
        })
    }

    /// Memory budget for all loaded shards together, in bytes.
    pub fn with_memory_budget_bytes(mut self, bytes: usize) -> Self {
        self.budget_bytes = bytes;
        self
    }

    /// Memory budget in mebibytes, as `[storage] memory_budget_mb` gives it. A budget smaller
    /// than one shard's indexes does not shrink anything; it only makes the pool reload on every
    /// request, so size it above the working set.
    pub fn with_memory_budget_mb(self, mb: usize) -> Self {
        self.with_memory_budget_bytes(mb.saturating_mul(1 << 20))
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
    }

    pub fn options(&self) -> &ShardOptions {
        &self.opts
    }

    /// Where `workspace`'s log, manifest and index files live.
    pub fn shard_dir(&self, workspace: Uuid) -> PathBuf {
        self.dir.join(workspace.to_string())
    }

    /// The shard for `workspace`, opened from disk or created empty, then evicts other shards as
    /// needed to get back under budget. The returned `Arc` pins it for as long as it is held.
    pub fn get(&self, workspace: Uuid) -> Result<Arc<Shard>> {
        if let Some(shard) = self.peek(workspace) {
            return Ok(shard);
        }

        // Slow path. The gate is taken from the map and the pool lock released before the shard
        // is opened, so this workspace's open blocks only callers who want this workspace.
        let gate = {
            let mut inner = self.lock();
            if let Some(shard) = inner.touch(workspace) {
                return Ok(shard);
            }
            inner
                .opening
                .entry(workspace)
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let opening = gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let opened = self.open_and_insert(workspace);

        // Both must happen whether or not the open worked: on failure the gate has to go so the
        // next caller can retry, and a waiter that already holds it keeps it alive until then.
        drop(opening);
        self.retire_gates();

        let shard = opened?;
        // A shard that could not be evicted is not this request's problem: the one it asked for
        // is open and usable, and the next pass tries again.
        if let Err(e) = self.enforce_budget() {
            warn!(
                "could not get back under budget after loading {}: {:#}",
                workspace, e
            );
        }
        Ok(shard)
    }

    /// The shard for `workspace` only when it already has a directory. Reads use this so a
    /// request naming a workspace that was never written does not leave an empty shard behind;
    /// only an insert should bring one into existence.
    pub fn get_existing(&self, workspace: Uuid) -> Result<Option<Arc<Shard>>> {
        if let Some(shard) = self.peek(workspace) {
            return Ok(Some(shard));
        }
        if !self.shard_dir(workspace).is_dir() {
            return Ok(None);
        }
        self.get(workspace).map(Some)
    }

    /// The shard for `workspace` only if it is already loaded. Bumps its recency.
    pub fn peek(&self, workspace: Uuid) -> Option<Arc<Shard>> {
        self.lock().touch(workspace)
    }

    pub fn is_loaded(&self, workspace: Uuid) -> bool {
        self.lock().shards.contains_key(&workspace)
    }

    /// Every loaded shard, for a caller that walks them all — the timer thread's compaction pass,
    /// say. Holding the returned `Arc`s pins those shards against eviction.
    pub fn loaded(&self) -> Vec<(Uuid, Arc<Shard>)> {
        self.lock()
            .shards
            .iter()
            .map(|(ws, entry)| (*ws, entry.shard.clone()))
            .collect()
    }

    /// Snapshots `workspace`'s shard and unloads it, unless a caller still holds it. Returns
    /// whether it was unloaded.
    pub fn evict(&self, workspace: Uuid) -> Result<bool> {
        let Some(shard) = self.claim_for_eviction(workspace) else {
            return Ok(false);
        };
        self.snapshot_and_remove(workspace, shard)
    }

    /// Evicts least-recently-used, unheld shards until the loaded set fits the budget. [`get`]
    /// does this itself; call it after bulk writes grew shards in place. Returns how many shards
    /// were evicted.
    ///
    /// [`get`]: Self::get
    pub fn enforce_budget(&self) -> Result<usize> {
        let mut evicted = 0;
        // A shard someone grabbed while we were snapshotting it stays loaded. Skipping it keeps
        // the loop from picking the same victim forever.
        let mut skip: HashSet<Uuid> = HashSet::new();
        loop {
            let victim = {
                let mut inner = self.lock();
                inner.refresh_bytes();
                if inner.total_bytes() <= self.budget_bytes {
                    return Ok(evicted);
                }
                inner
                    .shards
                    .iter()
                    .filter(|(ws, entry)| {
                        !skip.contains(*ws) && Arc::strong_count(&entry.shard) == 1
                    })
                    .min_by_key(|(_, entry)| entry.last_used)
                    .map(|(ws, entry)| (*ws, entry.shard.clone()))
            };
            // Everything left is either held by a caller or was grabbed mid-eviction.
            let Some((workspace, shard)) = victim else {
                return Ok(evicted);
            };
            if self.snapshot_and_remove(workspace, shard)? {
                evicted += 1;
            } else {
                skip.insert(workspace);
            }
        }
    }

    /// Snapshots every loaded shard, so the next open reads its indexes back instead of
    /// rebuilding them. A shard with nothing new in its log costs one lock and returns.
    ///
    /// Every shard is attempted even when one fails; the first error is returned afterwards.
    pub fn snapshot_all(&self) -> Result<usize> {
        self.for_each_loaded("snapshot", |shard| shard.snapshot().map(|_| ()))
    }

    /// Puts every loaded shard's acknowledged writes on the device.
    pub fn fsync_all(&self) -> Result<usize> {
        self.for_each_loaded("fsync", |shard| shard.fsync())
    }

    /// Rewrites the sealed segments of every loaded shard whose dead records take up more than
    /// `ratio` of them. Returns how many were compacted.
    pub fn compact_if_needed(&self, ratio: f64) -> Result<usize> {
        let mut compacted = 0;
        let mut first_error = None;
        for (workspace, shard) in self.loaded() {
            match shard.compact_if_needed(ratio) {
                Ok(true) => compacted += 1,
                Ok(false) => {}
                Err(e) => {
                    warn!("shard {}: compaction failed: {:#}", workspace, e);
                    first_error.get_or_insert(e);
                }
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(compacted),
        }
    }

    pub fn stats(&self) -> PoolStats {
        let mut inner = self.lock();
        inner.refresh_bytes();
        PoolStats {
            loaded: inner.shards.len(),
            memory_bytes: inner.total_bytes(),
            budget_bytes: self.budget_bytes,
        }
    }

    /// Every workspace with a directory under the pool's root, loaded or not, sorted. A name that
    /// is not a uuid is something else's and is ignored.
    pub fn list_workspaces(&self) -> Result<Vec<Uuid>> {
        let mut workspaces = Vec::new();
        let entries =
            fs::read_dir(&self.dir).with_context(|| format!("reading {}", self.dir.display()))?;
        for entry in entries {
            let entry = entry.with_context(|| format!("reading {}", self.dir.display()))?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            match entry
                .file_name()
                .to_str()
                .and_then(|n| Uuid::parse_str(n).ok())
            {
                Some(workspace) => workspaces.push(workspace),
                None => debug!(
                    "ignoring {}: not a workspace directory",
                    entry.path().display()
                ),
            }
        }
        workspaces.sort();
        Ok(workspaces)
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// A panic under the pool lock leaves the map itself intact — the shards it holds do their own
    /// recovery — so the lock is taken rather than propagated. Refusing every later request would
    /// be the larger failure.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|poisoned| {
            warn!(
                "shard pool {} recovered a poisoned lock",
                self.dir.display()
            );
            poisoned.into_inner()
        })
    }

    /// Opens the shard without the pool lock, then publishes it. Called with the workspace's gate
    /// held, so it cannot race another open of the same directory.
    fn open_and_insert(&self, workspace: Uuid) -> Result<Arc<Shard>> {
        // Whoever held the gate before us may have done the work already.
        if let Some(shard) = self.peek(workspace) {
            return Ok(shard);
        }

        let dir = self.shard_dir(workspace);
        let shard = Arc::new(
            Shard::open(&dir, self.opts.clone())
                .with_context(|| format!("opening shard for workspace {}", workspace))?,
        );
        // Nothing else can reach the shard yet, so this measurement never waits.
        let bytes = shard.stats().index_memory_bytes;
        info!(
            "workspace {}: shard loaded ({}, {} bytes resident)",
            workspace,
            if shard.loaded_from_snapshot() {
                "from snapshot"
            } else {
                "rebuilt"
            },
            bytes
        );

        let mut inner = self.lock();
        inner.tick += 1;
        let last_used = inner.tick;
        inner.shards.insert(
            workspace,
            Entry {
                shard: shard.clone(),
                last_used,
                bytes,
            },
        );
        Ok(shard)
    }

    /// Drops the gates no thread is holding. One still held keeps its entry, which is what stops
    /// a second opener from creating a rival gate while an open is in flight.
    fn retire_gates(&self) {
        self.lock()
            .opening
            .retain(|_, gate| Arc::strong_count(gate) > 1);
    }

    /// Takes a clone of an unheld shard, leaving it in the map. `None` when it is absent or in
    /// use.
    fn claim_for_eviction(&self, workspace: Uuid) -> Option<Arc<Shard>> {
        let inner = self.lock();
        let entry = inner.shards.get(&workspace)?;
        (Arc::strong_count(&entry.shard) == 1).then(|| entry.shard.clone())
    }

    /// Snapshots a claimed shard and then unloads it, provided nobody started using it in the
    /// meantime. The entry stays in the map for the whole snapshot: dropping it first would let a
    /// concurrent `get` open a second `Shard` on the same directory.
    fn snapshot_and_remove(&self, workspace: Uuid, shard: Arc<Shard>) -> Result<bool> {
        if let Err(e) = shard.snapshot() {
            // Keep it loaded rather than lose the writes the snapshot did not capture.
            warn!(
                "workspace {}: not evicting, its snapshot failed: {:#}",
                workspace, e
            );
            return Err(e);
        }

        let mut inner = self.lock();
        let Some(entry) = inner.shards.get(&workspace) else {
            return Ok(false);
        };
        // The map's reference plus ours. Anything more means a caller took it while we were
        // writing, and it stays.
        if Arc::strong_count(&entry.shard) > 2 {
            debug!("workspace {}: not evicting, it is in use", workspace);
            return Ok(false);
        }
        inner.shards.remove(&workspace);
        debug!("workspace {}: evicted", workspace);
        Ok(true)
    }

    fn for_each_loaded(&self, what: &str, op: impl Fn(&Shard) -> Result<()>) -> Result<usize> {
        let mut done = 0;
        let mut first_error = None;
        for (workspace, shard) in self.loaded() {
            match op(&shard) {
                Ok(()) => done += 1,
                Err(e) => {
                    warn!("shard {}: {} failed: {:#}", workspace, what, e);
                    first_error.get_or_insert(e);
                }
            }
        }
        match first_error {
            Some(e) => Err(e),
            None => Ok(done),
        }
    }
}

impl Inner {
    fn touch(&mut self, workspace: Uuid) -> Option<Arc<Shard>> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.shards.get_mut(&workspace)?;
        entry.last_used = tick;
        Some(entry.shard.clone())
    }

    /// Re-measures the shards that are free right now. A busy one keeps its previous figure.
    fn refresh_bytes(&mut self) {
        for entry in self.shards.values_mut() {
            if let Some(stats) = entry.shard.try_stats() {
                entry.bytes = stats.index_memory_bytes;
            }
        }
    }

    fn total_bytes(&self) -> usize {
        self.shards.values().map(|entry| entry.bytes).sum()
    }
}

impl std::fmt::Debug for ShardPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("ShardPool")
            .field("dir", &self.dir)
            .field("loaded", &stats.loaded)
            .field("memory_bytes", &stats.memory_bytes)
            .field("budget_bytes", &stats.budget_bytes)
            .finish()
    }
}
