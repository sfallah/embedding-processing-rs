//! Rebuilding the maps from the log.
//!
//! An insert writes its summaries and splits first and its `Doc` record last, and a document's
//! records are never split across segments, so replay buffers entities until their `Doc` record
//! arrives. Whatever is still buffered at the end was a torn insert and is dropped.

use crate::maps::{DocEntry, Loc, Maps, SplitEntry, SummaryEntry};
use crate::record::{decode, DecodeError, DeleteMeta, DocMeta, RecordKind, SplitMeta, SummaryMeta};
use anyhow::{anyhow, Context, Result};

#[derive(Debug, Default)]
pub struct SegmentScan {
    /// Records applied to the maps.
    pub applied: usize,
    /// Records skipped because the snapshot already covers them.
    pub skipped: usize,
    /// Offset just past the last well-formed record.
    pub good_bytes: u64,
    /// Offset of the first malformed record, if the segment has a torn tail.
    pub torn_at: Option<u64>,
    pub first_seq: Option<u64>,
    pub last_seq: u64,
}

/// One entity waiting for its document's `Doc` record.
#[derive(Debug)]
enum Pending {
    Split {
        split_id: u64,
        doc_id: u64,
        summary_ids: Vec<u64>,
        loc: Loc,
    },
    Summary {
        summary_id: u64,
        doc_id: u64,
        split_id: u64,
        loc: Loc,
    },
}

impl Pending {
    fn doc_id(&self) -> u64 {
        match self {
            Pending::Split { doc_id, .. } => *doc_id,
            Pending::Summary { doc_id, .. } => *doc_id,
        }
    }
}

/// Scan one segment's bytes, applying every record with `seq > snapshot_seq` to `maps`.
///
/// `allow_torn_tail` is true only for the active segment: a malformed record there means the
/// process died mid-append and everything from that offset on is discarded. The same in a sealed
/// segment is corruption and an error.
pub fn scan_segment(
    bytes: &[u8],
    segment_id: u32,
    snapshot_seq: u64,
    maps: &mut Maps,
    allow_torn_tail: bool,
) -> Result<SegmentScan> {
    let mut scan = SegmentScan::default();
    let mut pending: Vec<Pending> = Vec::new();
    let mut offset: usize = 0;

    while offset < bytes.len() {
        let view = match decode(&bytes[offset..]) {
            Ok(view) => view,
            Err(e) => {
                if allow_torn_tail && is_torn_tail(&e, &bytes[offset..]) {
                    scan.torn_at = Some(offset as u64);
                    break;
                }
                return Err(anyhow!(
                    "segment {} is corrupt at offset {}: {}",
                    segment_id,
                    offset,
                    e
                ));
            }
        };

        let loc = Loc {
            segment: segment_id,
            offset: offset as u64,
            len: view.len,
        };
        if scan.first_seq.is_none() {
            scan.first_seq = Some(view.seq);
        }
        scan.last_seq = view.seq;

        if view.seq > snapshot_seq {
            apply(view.kind, view.seq, view.meta, loc, maps, &mut pending)
                .with_context(|| format!("segment {} at offset {}", segment_id, offset))?;
            scan.applied += 1;
        } else {
            scan.skipped += 1;
        }

        offset += view.len as usize;
        scan.good_bytes = offset as u64;
    }

    // Anything still buffered belonged to an insert that never reached its `Doc` record.
    if !pending.is_empty() {
        tracing::warn!(
            "segment {}: dropping {} records from a torn insert",
            segment_id,
            pending.len()
        );
    }
    Ok(scan)
}

/// Is this bad record a torn write rather than corruption?
///
/// A record that runs past the end of the file can only be a torn write. A record that is fully
/// present but fails its checksum is only treated as one when nothing follows it: if there is a
/// complete record after it, the damage is in the middle of the log and discarding everything
/// from there on would silently destroy good documents, so it is reported instead.
fn is_torn_tail(e: &DecodeError, rest: &[u8]) -> bool {
    match e {
        DecodeError::Truncated { .. } => true,
        DecodeError::BadCrc { .. }
        | DecodeError::BadLength(_)
        | DecodeError::BadKind(_)
        | DecodeError::BadDtype(_) => match crate::record::peek_len(rest) {
            // The frame claims a length that reaches the end of the file: nothing follows.
            Some(len) => len as usize >= rest.len(),
            None => true,
        },
    }
}

fn apply(
    kind: RecordKind,
    seq: u64,
    meta: &[u8],
    loc: Loc,
    maps: &mut Maps,
    pending: &mut Vec<Pending>,
) -> Result<()> {
    match kind {
        RecordKind::Split => {
            let m: SplitMeta = rmp_serde::from_slice(meta)?;
            pending.push(Pending::Split {
                split_id: m.split_id,
                doc_id: m.doc_id,
                summary_ids: m.summary_ids,
                loc,
            });
        }
        RecordKind::Summary => {
            let m: SummaryMeta = rmp_serde::from_slice(meta)?;
            pending.push(Pending::Summary {
                summary_id: m.summary_id,
                doc_id: m.doc_id,
                split_id: m.split_id,
                loc,
            });
        }
        RecordKind::DeleteDoc => {
            let m: DeleteMeta = rmp_serde::from_slice(meta)?;
            // Accounting is carried in the manifest, so the segment id here is irrelevant.
            maps.remove_doc(m.doc_id, u32::MAX);
            pending.retain(|p| p.doc_id() != m.doc_id);
        }
        RecordKind::Doc => {
            let m: DocMeta = rmp_serde::from_slice(meta)?;
            commit_doc(m, seq, loc, maps, pending);
        }
    }
    Ok(())
}

/// Move this document's buffered entities into the maps and record the document itself.
fn commit_doc(meta: DocMeta, seq: u64, loc: Loc, maps: &mut Maps, pending: &mut Vec<Pending>) {
    let doc_id = meta.doc_id;

    // Replacing a document that is already present: drop what it owned first. A well-formed
    // insert wrote a DeleteDoc before its records, so this is belt and braces.
    if maps.docs.contains_key(&doc_id) {
        maps.remove_doc(doc_id, u32::MAX);
    }

    let mut split_summary_ids: Vec<u64> = Vec::new();
    let mut mine = Vec::new();
    let mut i = 0;
    while i < pending.len() {
        if pending[i].doc_id() == doc_id {
            mine.push(pending.remove(i));
        } else {
            i += 1;
        }
    }

    for entry in mine {
        match entry {
            Pending::Split {
                split_id,
                doc_id,
                summary_ids,
                loc,
            } => {
                split_summary_ids.extend(summary_ids);
                maps.splits.insert(split_id, SplitEntry { doc_id, loc });
            }
            Pending::Summary {
                summary_id,
                doc_id,
                split_id,
                loc,
            } => {
                maps.summaries.insert(
                    summary_id,
                    SummaryEntry {
                        doc_id,
                        split_id,
                        loc,
                    },
                );
            }
        }
    }

    let in_doc_list: std::collections::HashSet<u64> =
        meta.summary_ids.iter().flatten().copied().collect();
    let extra_summary_ids = dedup_preserving_order(
        split_summary_ids
            .into_iter()
            .filter(|id| !in_doc_list.contains(id))
            .collect(),
    );

    maps.docs.insert(
        doc_id,
        DocEntry {
            url: meta.url,
            split_ids: meta.split_ids,
            summary_ids: meta.summary_ids,
            extra_summary_ids,
            seq,
            loc,
        },
    );
}

fn dedup_preserving_order(ids: Vec<u64>) -> Vec<u64> {
    let mut seen = std::collections::HashSet::new();
    ids.into_iter().filter(|id| seen.insert(*id)).collect()
}
