//! Compaction: rewrite the sealed segments, keeping only live records.
//!
//! Every live record of a document sits in one segment — an insert never straddles a seal, because
//! sealing only happens at insert boundaries — so a document is copied as one group, its summaries
//! and splits before its `Doc` record, and the compacted segment replays exactly like the log it
//! replaces. Sequence numbers are carried over unchanged.
//!
//! The rewrite reads only immutable sealed segments, so it could run off the shard's write lock
//! and take the lock just for the swap (gap G7). It is synchronous here; the shard in step 3 is
//! the right place to move the read phase off the lock, since it owns the lock.

use crate::manifest::SegmentInfo;
use crate::maps::Loc;
use crate::meta_snapshot;
use crate::record::decode;
use crate::segment::{segment_path, ActiveSegment};
use crate::store::Store;
use anyhow::{anyhow, Result};
use std::collections::HashSet;
use std::fs;
use tracing::info;

pub fn compact(store: &mut Store) -> Result<()> {
    let sealed_ids: HashSet<u32> = store.sealed.keys().copied().collect();
    if sealed_ids.is_empty() {
        return Ok(());
    }

    // Claim the id before any file exists under it. A failure part way leaves a partial segment,
    // which is removed here and would be removed at the next open anyway; without claiming the id
    // first, a retry would find that file in the way and fail for good.
    let new_id = store.manifest.next_segment_id;
    store.manifest.next_segment_id += 1;
    match compact_into(store, new_id, &sealed_ids) {
        Ok(()) => Ok(()),
        Err(e) => {
            let partial = segment_path(&store.dir, new_id);
            if partial.exists() {
                if let Err(rm) = fs::remove_file(&partial) {
                    tracing::warn!("could not remove partial segment {}: {}", partial.display(), rm);
                }
            }
            Err(e)
        }
    }
}

fn compact_into(store: &mut Store, new_id: u32, sealed_ids: &HashSet<u32>) -> Result<()> {

    // Documents whose records live in a sealed segment, in a deterministic order.
    let mut doc_ids: Vec<u64> = store
        .maps
        .docs
        .iter()
        .filter(|(_, e)| sealed_ids.contains(&e.loc.segment))
        .map(|(id, _)| *id)
        .collect();
    doc_ids.sort_unstable();

    let mut writer = ActiveSegment::open(&store.dir, new_id)?;
    if writer.len != 0 {
        return Err(anyhow!(
            "segment {} already exists and is not empty",
            new_id
        ));
    }

    // (kind marker, id, new location) for every record copied.
    let mut new_split_locs: Vec<(u64, Loc)> = Vec::new();
    let mut new_summary_locs: Vec<(u64, Loc)> = Vec::new();
    let mut new_doc_locs: Vec<(u64, Loc)> = Vec::new();
    let mut first_seq = u64::MAX;
    let mut last_seq = 0u64;

    for doc_id in &doc_ids {
        let entry = store
            .maps
            .docs
            .get(doc_id)
            .expect("document id came from the map");
        let doc_loc = entry.loc;
        let summary_ids: Vec<u64> = entry.owned_summary_ids().collect();
        let split_ids = entry.split_ids.clone();

        for summary_id in &summary_ids {
            let Some(sum) = store.maps.summaries.get(summary_id) else {
                continue;
            };
            let loc = copy_record(store, &mut writer, sum.loc, &mut first_seq, &mut last_seq)?;
            new_summary_locs.push((*summary_id, loc));
        }
        for split_id in &split_ids {
            let Some(split) = store.maps.splits.get(split_id) else {
                continue;
            };
            let loc = copy_record(store, &mut writer, split.loc, &mut first_seq, &mut last_seq)?;
            new_split_locs.push((*split_id, loc));
        }
        let loc = copy_record(store, &mut writer, doc_loc, &mut first_seq, &mut last_seq)?;
        new_doc_locs.push((*doc_id, loc));
    }

    let bytes_written = writer.len;
    if bytes_written == 0 {
        // Nothing live in the sealed segments: drop them outright.
        fs::remove_file(segment_path(&store.dir, new_id)).ok();
        return drop_sealed(store);
    }

    let info = SegmentInfo {
        id: new_id,
        bytes: bytes_written,
        first_seq: if first_seq == u64::MAX { 0 } else { first_seq },
        last_seq,
    };
    let sealed_segment = writer.seal()?;

    // The snapshot describes the old locations, so it has to go before the manifest starts
    // pointing at the new segment. A crash between the two only costs a full replay.
    meta_snapshot::remove(&store.dir)?;
    store.manifest.snapshot_seq = 0;

    let old_ids: Vec<u32> = store.manifest.sealed.iter().map(|s| s.id).collect();
    store.manifest.sealed = vec![info];
    store.manifest.tombstone_bytes = 0;
    store.manifest.store(&store.dir)?;

    for (id, loc) in new_split_locs {
        if let Some(e) = store.maps.splits.get_mut(&id) {
            e.loc = loc;
        }
    }
    for (id, loc) in new_summary_locs {
        if let Some(e) = store.maps.summaries.get_mut(&id) {
            e.loc = loc;
        }
    }
    for (id, loc) in new_doc_locs {
        if let Some(e) = store.maps.docs.get_mut(&id) {
            e.loc = loc;
        }
    }

    store.sealed.clear();
    store.sealed.insert(new_id, sealed_segment);
    for id in old_ids {
        let path = segment_path(&store.dir, id);
        if let Err(e) = fs::remove_file(&path) {
            tracing::warn!("could not remove compacted segment {}: {}", path.display(), e);
        }
    }

    // Take a fresh snapshot so the next restart is still cheap.
    store.snapshot_meta()?;
    info!(
        "compacted {} documents into segment {} ({} bytes)",
        doc_ids.len(),
        new_id,
        bytes_written
    );
    Ok(())
}

/// Copy one record's bytes verbatim, keeping its sequence number.
fn copy_record(
    store: &Store,
    writer: &mut ActiveSegment,
    loc: Loc,
    first_seq: &mut u64,
    last_seq: &mut u64,
) -> Result<Loc> {
    let bytes = store.read_record(loc)?;
    let seq = decode(&bytes)
        .map_err(|e| anyhow!("compaction read a bad record at {:?}: {}", loc, e))?
        .seq;
    let offset = writer.append(&bytes, seq)?;
    *first_seq = (*first_seq).min(seq);
    *last_seq = (*last_seq).max(seq);
    Ok(Loc {
        segment: writer.id,
        offset,
        len: bytes.len() as u32,
    })
}

/// Drop every sealed segment without writing a replacement, because nothing in them is live.
fn drop_sealed(store: &mut Store) -> Result<()> {
    let old_ids: Vec<u32> = store.manifest.sealed.iter().map(|s| s.id).collect();
    meta_snapshot::remove(&store.dir)?;
    store.manifest.snapshot_seq = 0;
    store.manifest.sealed = Vec::new();
    store.manifest.tombstone_bytes = 0;
    store.manifest.store(&store.dir)?;
    store.sealed.clear();
    for id in old_ids {
        let path = segment_path(&store.dir, id);
        if let Err(e) = fs::remove_file(&path) {
            tracing::warn!("could not remove segment {}: {}", path.display(), e);
        }
    }
    store.snapshot_meta()?;
    Ok(())
}
