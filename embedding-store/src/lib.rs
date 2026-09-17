//! Per-workspace record store for the lean-storage design.
//!
//! One directory per workspace holds an append-only record log split into segments, a manifest,
//! and derived files that let a restart skip the log. Entities are stored one record each — a
//! document, a split, a summary — so serving one split's text deserialises one small record
//! rather than a whole document blob (gap G1).
//!
//! Durability model: every record is `len | crc | seq | kind | dtype | meta_len | meta | vector`.
//! An insert writes its summaries and splits first and its `Doc` record last, so a torn insert
//! replays as if it never happened. A torn tail in the active segment is truncated at the first
//! bad record; a bad record in a sealed segment is an error.
//!
//! Vectors are raw f16 by default (decision D1): the usearch index is f16 already, so nothing
//! searchable is lost, and the log is roughly 40% smaller than msgpack `Vec<f32>` would make it.
//! The record header carries a dtype byte, so f32 stays possible per record.

pub mod compact;
pub mod manifest;
pub mod maps;
pub mod meta_snapshot;
/// Reading a store written by the previous storage layer. Feature-gated: it is the only
/// thing here that depends on the `rocksdb` crate.
#[cfg(feature = "rocksdb-migration")]
pub mod migrate;
pub mod pool;
pub mod record;
pub mod replay;
pub mod segment;
pub mod shard;
pub mod store;
pub mod vector_index;

/// Make a rename durable. Writing a temp file and renaming it only guarantees "old or new" once
/// the directory entry itself has reached the device.
pub(crate) fn fsync_dir(dir: &std::path::Path) -> anyhow::Result<()> {
    let f = std::fs::File::open(dir)?;
    // Directory fsync is not supported everywhere; a failure here is not fatal to correctness of
    // the current process, only to the crash guarantee, so it is logged rather than propagated.
    if let Err(e) = f.sync_all() {
        tracing::debug!("could not fsync directory {}: {}", dir.display(), e);
    }
    Ok(())
}

pub mod prelude {
    pub use crate::manifest::{Manifest, SegmentInfo};
    pub use crate::maps::{DocEntry, Loc, Maps, SplitEntry, SummaryEntry};
    pub use crate::pool::{PoolStats, ShardPool};
    pub use crate::record::{RecordKind, VectorDtype};
    pub use crate::shard::{IndexResidency, Shard, ShardOptions, ShardStats};
    pub use crate::store::{EntityKind, Replaced, Store, StoreOptions};
    pub use crate::vector_index::VectorIndex;
}
