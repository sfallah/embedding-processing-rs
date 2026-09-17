use crate::utils::helpers::get_directory;
use anyhow::{bail, Result};
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

/// `[storage]`: where the per-workspace shards live, how much memory the loaded ones may hold
/// together, and how often the server puts them on the device.
///
/// One shard per workspace sits under [`dir`](Self::dir) as `<dir>/<workspace uuid>/`. The
/// defaults are raw f16 vectors in a 256 MB segment, a one-second fsync timer, and a budget
/// sized for roughly 750k splits.
#[derive(Debug, Deserialize, Clone, PartialEq)]
#[allow(unused)]
#[serde(deny_unknown_fields)]
pub struct StorageConfig {
    /// Root of the shard directories, relative to the working directory or absolute.
    #[serde(default = "default_dir")]
    pub dir: String,
    /// What the loaded shards' indexes may hold resident together. Shards over the budget are
    /// snapshotted and unloaded, least recently used first; one nobody is holding comes back on
    /// the next request by loading its saved indexes. Size it above the working set: a budget
    /// smaller than one shard does not shrink anything, it only reloads on every request.
    #[serde(default = "default_memory_budget_mb")]
    pub memory_budget_mb: usize,
    /// How often acknowledged writes are put on the device. `0` means after every write, which
    /// trades throughput for a window of zero.
    #[serde(default = "default_fsync_interval_ms")]
    pub fsync_interval_ms: u64,
    /// How often the indexes and the record maps are written out. A crash replays the log from
    /// the last snapshot, so this trades restart time for steady-state work.
    #[serde(default = "default_snapshot_interval_s")]
    pub snapshot_interval_s: u64,
    /// Seal the active segment once it passes this size. Sealed segments are immutable and
    /// mapped, and only they are compacted.
    #[serde(default = "default_segment_max_mb")]
    pub segment_max_mb: u64,
    /// Rewrite a shard's sealed segments once this fraction of them is dead records.
    #[serde(default = "default_compact_ratio")]
    pub compact_ratio: f64,
}

impl StorageConfig {
    /// The shard root as a path. A relative `dir` is taken from the working directory.
    pub fn storage_dir(&self) -> Result<PathBuf> {
        get_directory(&self.dir)
    }

    pub fn memory_budget_bytes(&self) -> usize {
        self.memory_budget_mb.saturating_mul(1 << 20)
    }

    pub fn segment_max_bytes(&self) -> u64 {
        self.segment_max_mb.saturating_mul(1 << 20)
    }

    /// How long the timer waits between fsync passes, or `None` when every write is to be synced
    /// as it happens. The two are different code paths, not two lengths of the same wait, which
    /// is why this is an `Option` rather than a zero `Duration`.
    pub fn fsync_interval(&self) -> Option<Duration> {
        (self.fsync_interval_ms > 0).then(|| Duration::from_millis(self.fsync_interval_ms))
    }

    pub fn snapshot_interval(&self) -> Duration {
        Duration::from_secs(self.snapshot_interval_s)
    }

    /// Rejects settings that would make the server misbehave rather than merely perform badly:
    /// a budget of nothing, a segment of nothing, or a compaction threshold outside `(0, 1]`.
    pub fn validate(&self) -> Result<()> {
        if self.dir.is_empty() {
            bail!("[storage] dir must not be empty");
        }
        if self.memory_budget_mb == 0 {
            bail!("[storage] memory_budget_mb must be greater than 0");
        }
        if self.segment_max_mb == 0 {
            bail!("[storage] segment_max_mb must be greater than 0");
        }
        if self.snapshot_interval_s == 0 {
            bail!("[storage] snapshot_interval_s must be greater than 0");
        }
        if !(self.compact_ratio > 0.0 && self.compact_ratio <= 1.0) {
            bail!(
                "[storage] compact_ratio must be in (0, 1], not {}",
                self.compact_ratio
            );
        }
        Ok(())
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        StorageConfig {
            dir: default_dir(),
            memory_budget_mb: default_memory_budget_mb(),
            fsync_interval_ms: default_fsync_interval_ms(),
            snapshot_interval_s: default_snapshot_interval_s(),
            segment_max_mb: default_segment_max_mb(),
            compact_ratio: default_compact_ratio(),
        }
    }
}

fn default_dir() -> String {
    "storage".to_string()
}

/// About 750k splits at 768 dimensions and f16, counting each split's summaries.
fn default_memory_budget_mb() -> usize {
    4096
}

fn default_fsync_interval_ms() -> u64 {
    1000
}

fn default_snapshot_interval_s() -> u64 {
    300
}

/// The store's own default, so the two agree unless the operator says otherwise.
fn default_segment_max_mb() -> u64 {
    256
}

fn default_compact_ratio() -> f64 {
    0.3
}
