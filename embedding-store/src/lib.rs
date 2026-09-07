//! Per-workspace record store for the lean-storage design (`PLAN-lean-storage.md`, step 2).
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
pub mod record;
pub mod replay;
pub mod segment;
pub mod store;

pub mod prelude {
    pub use crate::manifest::{Manifest, SegmentInfo};
    pub use crate::maps::{DocEntry, Loc, Maps, SplitEntry, SummaryEntry};
    pub use crate::record::{RecordKind, VectorDtype};
    pub use crate::store::{EntityKind, Replaced, Store, StoreOptions};
}
