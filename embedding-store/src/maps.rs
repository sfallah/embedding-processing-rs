//! The in-memory maps: where every live entity's record is, and the membership lists.
//!
//! These answer the document filter without a third `embed_id -> doc_id` map (gap G6): split ids
//! are keys of `splits`, summary ids are keys of `summaries`, and each index only ever returns
//! ids of its own kind.

use std::collections::HashMap;

/// Where one record lives. `segment` is a segment id, not a position in any list, so a `Loc`
/// stays valid when segments are sealed; only compaction rewrites them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Loc {
    pub segment: u32,
    pub offset: u64,
    pub len: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocEntry {
    pub url: String,
    /// In split order.
    pub split_ids: Vec<u64>,
    /// The document's own summary list, in recorded order. `None` and `Some(vec![])` are kept
    /// apart so a `DocumentDto` round-trips exactly.
    pub summary_ids: Option<Vec<u64>>,
    /// Summaries this document owns that are not in its own list, i.e. those that appear only in
    /// a split's list. Under decision D7 the document list is the union of the split lists and
    /// this is empty, but the storage does not assume that: a delete has to reach every summary
    /// the document owns without scanning the whole map or re-reading every split record.
    pub extra_summary_ids: Vec<u64>,
    pub seq: u64,
    pub loc: Loc,
}

impl DocEntry {
    /// Every summary id this document owns, the document's own list first.
    pub fn owned_summary_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.summary_ids
            .iter()
            .flatten()
            .chain(self.extra_summary_ids.iter())
            .copied()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitEntry {
    pub doc_id: u64,
    pub loc: Loc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SummaryEntry {
    pub doc_id: u64,
    pub split_id: u64,
    pub loc: Loc,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Maps {
    pub docs: HashMap<u64, DocEntry>,
    pub splits: HashMap<u64, SplitEntry>,
    pub summaries: HashMap<u64, SummaryEntry>,
}

impl Maps {
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty() && self.splits.is_empty() && self.summaries.is_empty()
    }

    /// Remove a document and everything it owns, returning the ids that went away so the caller
    /// can drop them from the indexes and account the freed bytes.
    ///
    /// Freed bytes are split by segment: only bytes in sealed segments are worth compacting, and
    /// counting the active segment's against the sealed total would trigger a rewrite that
    /// reclaims nothing.
    pub fn remove_doc(&mut self, doc_id: u64, active_segment: u32) -> Option<RemovedDoc> {
        let entry = self.docs.remove(&doc_id)?;
        let mut removed_splits = Vec::with_capacity(entry.split_ids.len());
        let mut removed_summaries = Vec::new();
        let mut sealed_bytes = 0u64;
        let mut active_bytes = 0u64;
        let mut account = |loc: &Loc, sealed: &mut u64, active: &mut u64| {
            if loc.segment == active_segment {
                *active += loc.len as u64;
            } else {
                *sealed += loc.len as u64;
            }
        };
        account(&entry.loc, &mut sealed_bytes, &mut active_bytes);

        for split_id in &entry.split_ids {
            if let Some(split) = self.splits.remove(split_id) {
                account(&split.loc, &mut sealed_bytes, &mut active_bytes);
                removed_splits.push(*split_id);
            }
        }
        // A summary may be a member of the document list, of a split list, or of both. It is
        // stored once, so remove it once, walking the ids the document owns.
        for summary_id in entry.owned_summary_ids() {
            if let Some(summary) = self.summaries.remove(&summary_id) {
                account(&summary.loc, &mut sealed_bytes, &mut active_bytes);
                removed_summaries.push(summary_id);
            }
        }
        drop(account);

        Some(RemovedDoc {
            entry,
            split_ids: removed_splits,
            summary_ids: removed_summaries,
            sealed_bytes,
            active_bytes,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RemovedDoc {
    pub entry: DocEntry,
    pub split_ids: Vec<u64>,
    pub summary_ids: Vec<u64>,
    /// Bytes the removed records occupied in sealed segments: the compaction trigger.
    pub sealed_bytes: u64,
    /// Bytes they occupied in the segment still being appended to.
    pub active_bytes: u64,
}
