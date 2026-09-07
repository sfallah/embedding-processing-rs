//! The manifest: which segments exist, how far the derived files are caught up, and the model the
//! shard is committed to. Written temp-and-rename, so a crash leaves either the old or the new one.

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const MANIFEST_VERSION: u32 = 1;
pub const MANIFEST_FILE: &str = "manifest";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentInfo {
    pub id: u32,
    pub bytes: u64,
    /// Sequence numbers of the first and last record in the segment. Replay skips a whole segment
    /// when `last_seq <= snapshot_seq`, which is what keeps a snapshot restart cheap.
    pub first_seq: u64,
    pub last_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub model_id: u64,
    pub n_embd: u32,
    /// `VectorDtype` as stored in record headers.
    pub dtype: u8,
    /// Sealed segments in replay order, which is not necessarily id order after a compaction.
    pub sealed: Vec<SegmentInfo>,
    /// Id of the segment currently being appended to.
    pub active: u32,
    /// Id to hand to the next segment created.
    pub next_segment_id: u32,
    pub next_seq: u64,
    /// Every record with `seq <= snapshot_seq` is already in the derived files.
    pub snapshot_seq: u64,
    /// Bytes occupied by records that are no longer live, in sealed segments.
    pub tombstone_bytes: u64,
}

impl Manifest {
    pub fn new(model_id: u64, n_embd: u32, dtype: u8) -> Self {
        Manifest {
            version: MANIFEST_VERSION,
            model_id,
            n_embd,
            dtype,
            sealed: Vec::new(),
            active: 0,
            next_segment_id: 1,
            next_seq: 1,
            snapshot_seq: 0,
            tombstone_bytes: 0,
        }
    }

    pub fn path(dir: &Path) -> PathBuf {
        dir.join(MANIFEST_FILE)
    }

    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let path = Self::path(dir);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let manifest: Manifest = rmp_serde::from_slice(&bytes)
            .with_context(|| format!("decoding {}", path.display()))?;
        if manifest.version > MANIFEST_VERSION {
            return Err(anyhow!(
                "manifest {} was written by a newer version ({} > {})",
                path.display(),
                manifest.version,
                MANIFEST_VERSION
            ));
        }
        Ok(Some(manifest))
    }

    pub fn store(&self, dir: &Path) -> Result<()> {
        let path = Self::path(dir);
        let tmp = dir.join(format!("{}.tmp", MANIFEST_FILE));
        let bytes = rmp_serde::to_vec_named(self)?;
        fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
        // The bytes must be on disk before the rename makes them the manifest.
        {
            let f = fs::File::open(&tmp)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))?;
        Ok(())
    }

    /// Total bytes across sealed segments, the denominator of the compaction ratio.
    pub fn sealed_bytes(&self) -> u64 {
        self.sealed.iter().map(|s| s.bytes).sum()
    }

    pub fn segment_ids(&self) -> Vec<u32> {
        self.sealed.iter().map(|s| s.id).collect()
    }
}
