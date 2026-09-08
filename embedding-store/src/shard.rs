//! `Shard`: one workspace's records and the two indexes over them.
//!
//! A shard owns a `Store` and a usearch index per searchable entity kind, behind one lock. The
//! lock is what makes the log and the indexes agree: a writer holds it while both are changed, a
//! reader holds it for the search and the loads that follow.
//!
//! The log is the source of truth. An index is derived from it and can always be rebuilt, which
//! is what lets `open` fall back to a rebuild whenever the saved index files are not provably the
//! ones this log's snapshot describes.

use crate::maps::DocEntry;
use crate::record::VectorDtype;
use crate::store::{EntityKind, Replaced, Store, StoreOptions};
use crate::vector_index::{distance, VectorIndex};
use anyhow::{anyhow, Context, Result};
use embedding_common::config::IndexConfig;
use embedding_common::prelude::*;
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use tracing::{info, warn};

/// Records which log position the index files on disk were written at.
///
/// The manifest's `snapshot_seq` alone is not enough: the store takes a meta snapshot of its own
/// during compaction, which moves `snapshot_seq` forward without the index files being rewritten.
/// A saved index is only trusted when this file says it was written at exactly the sequence the
/// manifest now calls the snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexState {
    version: u32,
    seq: u64,
    splits: u64,
    summaries: u64,
}

const INDEX_STATE_VERSION: u32 = 1;
const INDEX_STATE_FILE: &str = "indexes.state";

impl IndexState {
    fn path(dir: &Path) -> PathBuf {
        dir.join(INDEX_STATE_FILE)
    }

    fn load(dir: &Path) -> Option<Self> {
        let path = Self::path(dir);
        let bytes = fs::read(&path).ok()?;
        let state: IndexState = rmp_serde::from_slice(&bytes).ok()?;
        (state.version == INDEX_STATE_VERSION).then_some(state)
    }

    fn store(&self, dir: &Path) -> Result<()> {
        let path = Self::path(dir);
        let tmp = dir.join(format!("{}.tmp", INDEX_STATE_FILE));
        fs::write(&tmp, rmp_serde::to_vec_named(self)?)
            .with_context(|| format!("writing {}", tmp.display()))?;
        fs::File::open(&tmp)?.sync_all()?;
        fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))?;
        crate::fsync_dir(dir)?;
        Ok(())
    }

    fn remove(dir: &Path) -> Result<()> {
        let path = Self::path(dir);
        if path.exists() {
            fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ShardOptions {
    pub model_id: u64,
    /// Dimensions, metric and HNSW parameters. Its `index_dir` is not used: a shard keeps its
    /// index files next to its own log.
    pub index: IndexConfig,
    pub segment_max_bytes: u64,
    /// Above this many candidates a document-filtered search walks the graph instead of scoring
    /// the candidates one by one. `filtered_search` was measured about five times slower than a
    /// plain search at 10% selectivity, and worse as the filter tightens.
    pub brute_force_max: usize,
}

impl ShardOptions {
    pub fn new(model_id: u64, index: IndexConfig) -> Self {
        let store_defaults = StoreOptions::new(model_id, index.dimensions);
        ShardOptions {
            model_id,
            index,
            segment_max_bytes: store_defaults.segment_max_bytes,
            brute_force_max: 4096,
        }
    }

    fn store_options(&self) -> StoreOptions {
        let mut opts = StoreOptions::new(self.model_id, self.index.dimensions);
        opts.segment_max_bytes = self.segment_max_bytes;
        opts
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardStats {
    pub docs: usize,
    pub splits: usize,
    pub summaries: usize,
    pub split_index_size: usize,
    pub summary_index_size: usize,
    /// Resident bytes in the two indexes. The maps and the mapped segments are not counted here.
    pub index_memory_bytes: usize,
    pub snapshot_seq: u64,
    pub next_seq: u64,
    pub sealed_bytes: u64,
    pub tombstone_bytes: u64,
}

struct ShardInner {
    store: Store,
    splits: VectorIndex,
    summaries: VectorIndex,
}

pub struct Shard {
    dir: PathBuf,
    opts: ShardOptions,
    inner: RwLock<ShardInner>,
    /// Whether `open` read the indexes back rather than rebuilding them from the log.
    from_snapshot: bool,
}

/// How a shard's indexes are held in memory.
///
/// A shard nobody is writing to does not need its graphs resident: usearch can answer searches
/// from the mapped file. `Viewed` is what the pool demotes a cold shard to instead of dropping it,
/// so its record maps stay and a later read costs nothing (decision D3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexResidency {
    /// Graphs in memory, writable.
    Loaded,
    /// Graphs mapped from the saved files, read-only. Falls back to `Loaded` when there is no
    /// saved index to map, because a rebuilt graph has nowhere to be mapped from.
    Viewed,
}

impl Shard {
    pub fn open(dir: impl AsRef<Path>, opts: ShardOptions) -> Result<Self> {
        Self::open_with(dir, opts, IndexResidency::Loaded)
    }

    /// Open, preferring `residency`. `Viewed` is only honoured when the saved index files are
    /// usable; otherwise the indexes are rebuilt and the shard comes back `Loaded`.
    pub fn open_with(
        dir: impl AsRef<Path>,
        opts: ShardOptions,
        residency: IndexResidency,
    ) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let store = Store::open(&dir, opts.store_options())?;
        let splits = VectorIndex::create(&dir, "splits", &opts.index)?;
        let summaries = VectorIndex::create(&dir, "summaries", &opts.index)?;

        let read_back = |index: &VectorIndex| match residency {
            IndexResidency::Loaded => index.load(),
            IndexResidency::Viewed => index.view(),
        };

        let from_snapshot = match snapshot_is_usable(&dir, &store, &splits, &summaries) {
            Ok(()) => match (read_back(&splits), read_back(&summaries)) {
                (Ok(()), Ok(())) => {
                    if splits.size() < store.split_count()
                        || summaries.size() < store.summary_count()
                    {
                        warn!(
                            "shard {}: saved indexes hold {}/{} vectors for {}/{} live records, rebuilding",
                            dir.display(),
                            splits.size(),
                            summaries.size(),
                            store.split_count(),
                            store.summary_count()
                        );
                        false
                    } else {
                        info!(
                            "shard {}: loaded indexes at seq {} ({} splits, {} summaries)",
                            dir.display(),
                            store.manifest().snapshot_seq,
                            splits.size(),
                            summaries.size()
                        );
                        true
                    }
                }
                (splits_result, summaries_result) => {
                    let reason = splits_result.err().or(summaries_result.err());
                    warn!(
                        "shard {}: could not load the saved indexes ({}), rebuilding",
                        dir.display(),
                        reason.map(|e| e.to_string()).unwrap_or_default()
                    );
                    false
                }
            },
            Err(reason) => {
                if !store.maps().is_empty() {
                    info!(
                        "shard {}: rebuilding the indexes from the log ({})",
                        dir.display(),
                        reason
                    );
                }
                false
            }
        };

        if !from_snapshot {
            // A half-loaded index has to go before it is filled from the log, and the state file
            // with it: it describes files that are about to stop matching what is on disk.
            for index in [&splits, &summaries] {
                index.remove_file()?;
            }
            IndexState::remove(&dir)?;
            rebuild(&store, &splits, &summaries)?;
        }

        Ok(Shard {
            dir,
            opts,
            inner: RwLock::new(ShardInner {
                store,
                splits,
                summaries,
            }),
            from_snapshot,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn options(&self) -> &ShardOptions {
        &self.opts
    }

    /// True when `open` read the indexes back instead of rebuilding them. The difference is the
    /// whole point of the snapshot path, so it is worth being able to assert on.
    pub fn loaded_from_snapshot(&self) -> bool {
        self.from_snapshot
    }

    pub fn residency(&self) -> IndexResidency {
        let inner = self.read();
        if inner.splits.is_viewed() || inner.summaries.is_viewed() {
            IndexResidency::Viewed
        } else {
            IndexResidency::Loaded
        }
    }

    /// Make the indexes writable again by reading in what is currently mapped. Cheap — about 2 ms
    /// per 8 MB — and a no-op on a shard that is already `Loaded`, which is why the write path can
    /// simply call it.
    pub fn promote(&self) -> Result<()> {
        self.write().promote()
    }

    /// Snapshot, then hand the graphs back to the page cache. The record maps stay resident, so a
    /// read still costs nothing; only the graphs go.
    ///
    /// The snapshot is what makes this safe: there has to be a file to map, and it has to describe
    /// this exact log.
    pub fn demote(&self) -> Result<()> {
        self.snapshot()?;
        let inner = self.write();
        if inner.splits.is_viewed() && inner.summaries.is_viewed() {
            return Ok(());
        }
        inner.splits.view()?;
        inner.summaries.view()?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // The lock
    // -----------------------------------------------------------------------

    /// A panic while the shard was being written leaves the lock poisoned. The log is the source
    /// of truth and a restart rebuilds whatever the maps got wrong, so a poisoned lock is taken
    /// rather than propagated: one bad request must not take every later request down with it.
    fn read(&self) -> RwLockReadGuard<'_, ShardInner> {
        self.inner.read().unwrap_or_else(|poisoned| {
            warn!("shard {} recovered a poisoned lock", self.dir.display());
            poisoned.into_inner()
        })
    }

    fn write(&self) -> RwLockWriteGuard<'_, ShardInner> {
        self.inner.write().unwrap_or_else(|poisoned| {
            warn!("shard {} recovered a poisoned lock", self.dir.display());
            poisoned.into_inner()
        })
    }

    // -----------------------------------------------------------------------
    // Writing
    // -----------------------------------------------------------------------

    /// Store a document and put its vectors in the indexes, replacing any document with the same
    /// id.
    ///
    /// The store goes first: it validates the document, and until it has accepted one there is
    /// nothing to index. An index update that fails afterwards leaves the log ahead of the
    /// indexes, which the next `open` repairs by rebuilding; the error is still returned, because
    /// the document is not searchable until then.
    pub fn insert(&self, doc: &DocumentDto) -> Result<Replaced> {
        let mut inner = self.write();
        // usearch refuses `add` on a mapped index, so a cold shard is read in before it is written
        // to. Nothing has been appended to the log at this point, so a failure here changes
        // nothing.
        inner.promote()?;
        let replaced = inner.store.insert(doc)?;

        for split_id in &replaced.split_ids {
            inner.splits.remove(*split_id)?;
        }
        for summary_id in &replaced.summary_ids {
            inner.summaries.remove(*summary_id)?;
        }

        inner.splits.reserve_for(doc.splits.len())?;
        for split in &doc.splits {
            let embedding = split
                .embedding
                .as_ref()
                .ok_or_else(|| anyhow!("split {} has no embedding", split.split_id))?;
            inner.splits.upsert(split.split_id, &embedding.embedding)?;
        }

        // A summary can be named by the document's list, by a split's list, or by both; it is one
        // entity with one vector, so it is indexed once.
        let mut seen = HashSet::new();
        for summary in doc
            .summaries
            .iter()
            .flatten()
            .chain(doc.splits.iter().flat_map(|split| split.summaries.iter()))
        {
            if !seen.insert(summary.summary_id) {
                continue;
            }
            let embedding = summary
                .embedding
                .as_ref()
                .ok_or_else(|| anyhow!("summary {} has no embedding", summary.summary_id))?;
            inner
                .summaries
                .upsert(summary.summary_id, &embedding.embedding)?;
        }

        Ok(replaced)
    }

    /// Remove a document, its splits and its summaries from the log and from both indexes.
    pub fn delete(&self, doc_id: u64) -> Result<bool> {
        let mut inner = self.write();
        // Nothing to do for a document this shard does not hold, and in particular no reason to
        // read the graphs in.
        if !inner.store.contains_doc(doc_id) {
            return Ok(false);
        }
        inner.promote()?;
        let Some(removed) = inner.store.delete(doc_id)? else {
            return Ok(false);
        };
        for split_id in &removed.split_ids {
            inner.splits.remove(*split_id)?;
        }
        for summary_id in &removed.summary_ids {
            inner.summaries.remove(*summary_id)?;
        }
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Searching
    // -----------------------------------------------------------------------

    /// The `top_k` nearest splits, ascending by distance.
    ///
    /// `doc_filter` restricts the search to those documents. `None` means the whole shard;
    /// `Some` of an empty set matches nothing, so a caller with no document restriction passes
    /// `None` rather than an empty set.
    pub fn search_splits(
        &self,
        query: &[f32],
        top_k: usize,
        doc_filter: Option<&HashSet<u64>>,
    ) -> Result<IndexMap<u64, f32>> {
        self.read().search(
            EntityKind::Split,
            query,
            top_k,
            doc_filter,
            self.opts.brute_force_max,
        )
    }

    /// The `top_k` nearest summaries, ascending by distance. Filtering as in `search_splits`.
    pub fn search_summaries(
        &self,
        query: &[f32],
        top_k: usize,
        doc_filter: Option<&HashSet<u64>>,
    ) -> Result<IndexMap<u64, f32>> {
        self.read().search(
            EntityKind::Summary,
            query,
            top_k,
            doc_filter,
            self.opts.brute_force_max,
        )
    }

    // -----------------------------------------------------------------------
    // Reading
    // -----------------------------------------------------------------------

    /// Summaries in the order asked for, each carrying the distance the search gave it.
    ///
    /// Ids this shard does not hold are skipped rather than reported: a search returns what the
    /// index knows, and an id can only be missing here if it was deleted between the two calls.
    pub fn load_summaries(
        &self,
        ids: &[u64],
        distances: Option<&IndexMap<u64, f32>>,
        with_embeddings: bool,
    ) -> Result<Vec<SummaryDto>> {
        let inner = self.read();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(mut summary) = inner.store.get_summary(*id, with_embeddings)? {
                summary.query_distance = distances.and_then(|d| d.get(id).copied());
                out.push(summary);
            }
        }
        Ok(out)
    }

    /// Splits in the order asked for.
    ///
    /// `direct_distances` holds the splits the query hit itself; a split that only got here
    /// through one of its summaries has no distance of its own. `hit_summaries`, when given,
    /// replaces each split's own summary list with the summaries the query hit — which is what a
    /// query returns, as against retrieval, where the whole list belongs to the answer.
    pub fn load_splits(
        &self,
        ids: &[u64],
        direct_distances: Option<&IndexMap<u64, f32>>,
        hit_summaries: Option<&IndexMap<u64, Vec<SummaryDto>>>,
        with_embeddings: bool,
    ) -> Result<Vec<SplitDto>> {
        let inner = self.read();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let split = match hit_summaries {
                Some(_) => inner
                    .store
                    .get_split_without_summaries(*id, with_embeddings)?,
                None => inner.store.get_split(*id, with_embeddings)?,
            };
            let Some(mut split) = split else {
                continue;
            };
            if let Some(hits) = hit_summaries {
                split.summaries = hits.get(id).cloned().unwrap_or_default();
            }
            split.query_distance = direct_distances.and_then(|d| d.get(id).copied());
            out.push(split);
        }
        Ok(out)
    }

    /// One document carrying the splits it was hit through, plus its own summary list in recorded
    /// order and without distances.
    pub fn load_doc(
        &self,
        doc_id: u64,
        splits: Vec<SplitDto>,
        with_embeddings: bool,
    ) -> Result<Option<DocumentDto>> {
        let inner = self.read();
        let Some(entry) = inner.store.maps().docs.get(&doc_id) else {
            return Ok(None);
        };
        let summaries = match &entry.summary_ids {
            Some(ids) => Some(inner.store.get_summaries(ids, with_embeddings)?),
            None => None,
        };
        Ok(Some(DocumentDto::new(
            doc_id, &entry.url, splits, summaries,
        )))
    }

    /// A whole document: every split with all of its summaries, plus the document's own list.
    pub fn get_doc(&self, doc_id: u64, with_embeddings: bool) -> Result<Option<DocumentDto>> {
        self.read().store.get_doc(doc_id, with_embeddings)
    }

    pub fn doc_id_for_url(&self, url: &str) -> Option<u64> {
        self.read().store.doc_id_for_url(url)
    }

    pub fn contains_doc(&self, doc_id: u64) -> bool {
        self.read().store.contains_doc(doc_id)
    }

    /// A copy of one document's membership lists, for a caller that needs the ids without the
    /// records behind them.
    pub fn doc_entry(&self, doc_id: u64) -> Option<DocEntry> {
        self.read().store.maps().docs.get(&doc_id).cloned()
    }

    // -----------------------------------------------------------------------
    // Durability and housekeeping
    // -----------------------------------------------------------------------

    /// Put every acknowledged write on the device. Cheap: it syncs the active segment only.
    pub fn fsync(&self) -> Result<()> {
        self.write().store.fsync()
    }

    /// Save both indexes and the store's maps, so the next `open` reads them instead of replaying
    /// the log and rebuilding the graphs.
    ///
    /// The order is what makes a crash safe. The log is synced first, then the index files are
    /// published with the sequence they cover, and only then does the manifest call that sequence
    /// the snapshot. A crash anywhere before the last step leaves a manifest that names an older
    /// snapshot, or none, and `open` rebuilds; it never loads index files the manifest disagrees
    /// with.
    pub fn snapshot(&self) -> Result<u64> {
        let mut inner = self.write();
        inner.store.fsync()?;

        let seq = inner.store.manifest().next_seq.saturating_sub(1);
        if seq == inner.store.manifest().snapshot_seq && IndexState::load(&self.dir).is_some() {
            return Ok(seq);
        }

        inner.splits.save_atomically()?;
        inner.summaries.save_atomically()?;
        IndexState {
            version: INDEX_STATE_VERSION,
            seq,
            splits: inner.splits.size() as u64,
            summaries: inner.summaries.size() as u64,
        }
        .store(&self.dir)?;

        let published = inner.store.snapshot_meta()?;
        if published != seq {
            // Nothing writes to the shard without the write lock, so the two sequences cannot
            // drift apart. If they ever did, the index files would claim a position they do not
            // hold, so the state file goes rather than the mismatch being left on disk.
            IndexState::remove(&self.dir)?;
            return Err(anyhow!(
                "shard {}: the log moved from seq {} to {} during a snapshot",
                self.dir.display(),
                seq,
                published
            ));
        }
        Ok(seq)
    }

    pub fn seal_active(&self) -> Result<()> {
        self.write().store.seal_active()
    }

    /// Rewrite the sealed segments when dead records take up more than `ratio` of them, then take
    /// a fresh snapshot.
    ///
    /// Compaction moves records, which resets the store's own snapshot; without the snapshot that
    /// follows, the index files on disk would describe a position the manifest no longer names
    /// and the next open would rebuild for no reason.
    pub fn compact_if_needed(&self, ratio: f64) -> Result<bool> {
        {
            let mut inner = self.write();
            if !inner.store.needs_compaction(ratio) {
                return Ok(false);
            }
            // Compaction ends in a snapshot, which writes the index files; a mapped index cannot
            // be the one it is written from, so it is read in first.
            inner.promote()?;
            inner.store.compact_if_needed(ratio)?;
        }
        self.snapshot()?;
        Ok(true)
    }

    pub fn stats(&self) -> ShardStats {
        Self::stats_of(&self.read())
    }

    /// [`stats`](Self::stats) when the shard is free, `None` when it is busy.
    ///
    /// The pool measures every loaded shard to enforce its memory budget. A blocking read there
    /// would put one shard's multi-second snapshot in front of every other workspace's requests,
    /// so the pool takes a stale figure over a wait.
    pub fn try_stats(&self) -> Option<ShardStats> {
        match self.inner.try_read() {
            Ok(inner) => Some(Self::stats_of(&inner)),
            Err(TryLockError::Poisoned(poisoned)) => Some(Self::stats_of(&poisoned.into_inner())),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    fn stats_of(inner: &ShardInner) -> ShardStats {
        let manifest = inner.store.manifest();
        ShardStats {
            docs: inner.store.doc_count(),
            splits: inner.store.split_count(),
            summaries: inner.store.summary_count(),
            split_index_size: inner.splits.size(),
            summary_index_size: inner.summaries.size(),
            index_memory_bytes: inner.splits.memory_bytes() + inner.summaries.memory_bytes(),
            snapshot_seq: manifest.snapshot_seq,
            next_seq: manifest.next_seq,
            sealed_bytes: manifest.sealed_bytes(),
            tombstone_bytes: manifest.tombstone_bytes,
        }
    }
}

impl ShardInner {
    /// Read in whichever graphs are only mapped. A no-op when both are already resident.
    fn promote(&self) -> Result<()> {
        for index in [&self.splits, &self.summaries] {
            if index.is_viewed() {
                index.load()?;
            }
        }
        Ok(())
    }

    fn index(&self, kind: EntityKind) -> &VectorIndex {
        match kind {
            EntityKind::Split => &self.splits,
            EntityKind::Summary => &self.summaries,
        }
    }

    fn search(
        &self,
        kind: EntityKind,
        query: &[f32],
        top_k: usize,
        doc_filter: Option<&HashSet<u64>>,
        brute_force_max: usize,
    ) -> Result<IndexMap<u64, f32>> {
        let index = self.index(kind);
        if top_k == 0 {
            return Ok(IndexMap::new());
        }
        let hits = match doc_filter {
            None => index.search(query, top_k)?,
            Some(doc_ids) => {
                let candidates = self.candidate_count(kind, doc_ids);
                if candidates == 0 {
                    Vec::new()
                } else if candidates <= brute_force_max {
                    self.brute_force(kind, query, top_k, doc_ids)?
                } else {
                    // The maps answer the filter directly: split ids are keys of one map and
                    // summary ids of the other, and each index only ever offers keys of its own
                    // kind, so no third id-to-document map is needed.
                    let maps = self.store.maps();
                    index.filtered_search(query, top_k, |key| match kind {
                        EntityKind::Split => maps
                            .splits
                            .get(&key)
                            .is_some_and(|entry| doc_ids.contains(&entry.doc_id)),
                        EntityKind::Summary => maps
                            .summaries
                            .get(&key)
                            .is_some_and(|entry| doc_ids.contains(&entry.doc_id)),
                    })?
                }
            }
        };
        Ok(hits.into_iter().collect())
    }

    /// How many entities of this kind the filtered documents hold, without listing them.
    fn candidate_count(&self, kind: EntityKind, doc_ids: &HashSet<u64>) -> usize {
        doc_ids
            .iter()
            .filter_map(|doc_id| self.store.maps().docs.get(doc_id))
            .map(|entry| match kind {
                EntityKind::Split => entry.split_ids.len(),
                EntityKind::Summary => {
                    entry.summary_ids.as_ref().map_or(0, |ids| ids.len())
                        + entry.extra_summary_ids.len()
                }
            })
            .sum()
    }

    /// Score every candidate directly from the log. Below a few thousand candidates this beats
    /// `filtered_search`, which pays for the graph walk and then throws most of it away.
    fn brute_force(
        &self,
        kind: EntityKind,
        query: &[f32],
        top_k: usize,
        doc_ids: &HashSet<u64>,
    ) -> Result<Vec<(u64, f32)>> {
        let metric = self.index(kind).metric();
        let mut candidates: Vec<u64> = Vec::new();
        for doc_id in doc_ids {
            let Some(entry) = self.store.maps().docs.get(doc_id) else {
                continue;
            };
            match kind {
                EntityKind::Split => candidates.extend(entry.split_ids.iter().copied()),
                EntityKind::Summary => candidates.extend(entry.owned_summary_ids()),
            }
        }

        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(candidates.len());
        for id in candidates {
            let scored_one = self.store.with_vector(kind, id, |bytes, dtype| {
                distance(metric, query, &crate::record::decode_vector(bytes, dtype))
            })?;
            if let Some(d) = scored_one {
                scored.push((id, d));
            }
        }
        // Ties are broken by id so that the same query over the same shard always answers with
        // the same list, whatever order the maps happened to hand the candidates over in.
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(top_k);
        Ok(scored)
    }
}

/// Are the index files on disk the ones this log's snapshot describes?
///
/// Both conditions matter. The state file has to name the sequence the manifest now calls the
/// snapshot, and the log must hold nothing beyond it: records replayed from a tail are in the
/// maps but not in a saved index, and there is no record of which entities they touched.
fn snapshot_is_usable(
    dir: &Path,
    store: &Store,
    splits: &VectorIndex,
    summaries: &VectorIndex,
) -> std::result::Result<(), String> {
    if !splits.exists() || !summaries.exists() {
        return Err("no saved indexes".to_string());
    }
    let Some(state) = IndexState::load(dir) else {
        return Err("the index state file is missing or unreadable".to_string());
    };
    let manifest = store.manifest();
    if state.seq != manifest.snapshot_seq {
        return Err(format!(
            "the indexes were saved at seq {} but the snapshot is at {}",
            state.seq, manifest.snapshot_seq
        ));
    }
    if manifest.snapshot_seq + 1 != manifest.next_seq {
        return Err(format!(
            "{} records were replayed after the snapshot at seq {}",
            manifest.next_seq - manifest.snapshot_seq - 1,
            manifest.snapshot_seq
        ));
    }
    Ok(())
}

/// Fill both indexes from the live records, adding the vectors as the log holds them.
fn rebuild(store: &Store, splits: &VectorIndex, summaries: &VectorIndex) -> Result<()> {
    splits.reserve_for(store.split_count())?;
    summaries.reserve_for(store.summary_count())?;
    for (kind, index) in [
        (EntityKind::Split, splits),
        (EntityKind::Summary, summaries),
    ] {
        store.for_each_live_vector(kind, |id, bytes, dtype| {
            if dtype == VectorDtype::None {
                return Err(anyhow!("record {} has no vector", id));
            }
            index.add_raw(id, bytes, dtype)
        })?;
    }
    Ok(())
}
