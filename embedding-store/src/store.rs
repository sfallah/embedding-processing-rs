//! `Store`: the record log plus the maps, one per workspace shard.
//!
//! The store owns durability and layout. It knows nothing about vector search: `Shard` wraps it
//! together with the two usearch indexes and a lock.

use crate::manifest::{Manifest, SegmentInfo};
use crate::maps::{DocEntry, Loc, Maps, RemovedDoc, SplitEntry, SummaryEntry};
use crate::meta_snapshot;
use crate::record::{
    decode, decode_vector, encode, encode_vector, DeleteMeta, DocMeta, RecordKind, SplitMeta,
    SummaryMeta, VectorDtype,
};
use crate::replay::scan_segment;
use crate::segment::{segment_path, ActiveSegment, SealedSegment};
use anyhow::{anyhow, Context, Result};
use embedding_common::prelude::*;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// Which of the two searchable entity kinds a call refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Split,
    Summary,
}

#[derive(Debug, Clone)]
pub struct StoreOptions {
    /// The model every embedding in this shard must come from.
    pub model_id: u64,
    pub n_embd: usize,
    pub dtype: VectorDtype,
    /// Seal the active segment once it passes this size.
    pub segment_max_bytes: u64,
}

impl StoreOptions {
    pub fn new(model_id: u64, n_embd: usize) -> Self {
        StoreOptions {
            model_id,
            n_embd,
            dtype: VectorDtype::F16,
            segment_max_bytes: 256 * 1024 * 1024,
        }
    }
}

/// What an insert displaced, so the caller can keep the indexes in step.
///
/// `split_ids` and `summary_ids` are the entities the replaced document owned that the new one
/// does **not**, so a caller can remove exactly these from its indexes with no risk of deleting a
/// vector the same insert has just written. Ids the new document reuses — which is the common case,
/// since ids are a deterministic hash of the document and its sequence numbers — are simply
/// overwritten and never appear here.
#[derive(Debug, Clone, Default)]
pub struct Replaced {
    pub existed: bool,
    pub split_ids: Vec<u64>,
    pub summary_ids: Vec<u64>,
}

/// What one document's records ended up as, before any of it reaches the maps.
struct Appended {
    replaces: bool,
    /// (summary_id, doc_id, split_id, location)
    summaries: Vec<(u64, u64, u64, Loc)>,
    /// (split_id, doc_id, location)
    splits: Vec<(u64, u64, Loc)>,
    doc_entry: DocEntry,
}

pub struct Store {
    pub(crate) dir: PathBuf,
    pub(crate) opts: StoreOptions,
    pub(crate) manifest: Manifest,
    pub(crate) sealed: BTreeMap<u32, SealedSegment>,
    pub(crate) active: ActiveSegment,
    pub(crate) maps: Maps,
    /// Dead bytes in the segment still being appended to. They only become worth compacting once
    /// that segment is sealed, at which point they move into `manifest.tombstone_bytes`.
    pub(crate) pending_tombstone_bytes: u64,
}

impl Store {
    // -----------------------------------------------------------------------
    // Opening
    // -----------------------------------------------------------------------

    pub fn open(dir: impl AsRef<Path>, opts: StoreOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

        let mut manifest = match Manifest::load(&dir)? {
            Some(m) => {
                if m.model_id != opts.model_id {
                    return Err(anyhow!(
                        "shard {} holds model {:#x}, refusing to open it for model {:#x}",
                        dir.display(),
                        m.model_id,
                        opts.model_id
                    ));
                }
                if m.n_embd as usize != opts.n_embd {
                    return Err(anyhow!(
                        "shard {} was written with {} dimensions, not {}",
                        dir.display(),
                        m.n_embd,
                        opts.n_embd
                    ));
                }
                if m.dtype != opts.dtype as u8 {
                    return Err(anyhow!(
                        "shard {} stores vectors as dtype {}, not {}",
                        dir.display(),
                        m.dtype,
                        opts.dtype as u8
                    ));
                }
                m
            }
            None => {
                let m = Manifest::new(opts.model_id, opts.n_embd as u32, opts.dtype as u8);
                m.store(&dir)?;
                m
            }
        };

        remove_orphan_segments(&dir, &manifest)?;

        let mut sealed = BTreeMap::new();
        for info in &manifest.sealed {
            let path = segment_path(&dir, info.id);
            sealed.insert(info.id, SealedSegment::open(&path, info.id)?);
        }
        let mut active = ActiveSegment::open(&dir, manifest.active)?;

        // The snapshot only counts if it is the one this manifest describes, and if every
        // location in it still fits inside the segments on disk. A snapshot that outruns the log
        // was published before a tail that never reached the device; trusting it would let replay
        // skip those records as already covered, find no torn tail to truncate, and let the next
        // append land on locations the maps still point at.
        let mut rejected_snapshot = None;
        let (mut maps, snapshot_seq) = match meta_snapshot::load(&dir) {
            Ok(Some((_maps, seq))) if seq != manifest.snapshot_seq => {
                rejected_snapshot = Some(format!(
                    "it is at seq {} but the manifest says {}",
                    seq, manifest.snapshot_seq
                ));
                (Maps::default(), 0)
            }
            Ok(Some((maps, _seq))) if !locations_fit(&maps, &sealed, &active) => {
                rejected_snapshot = Some("it points past the end of the log".to_string());
                (Maps::default(), 0)
            }
            Ok(Some((maps, seq))) => (maps, seq),
            Ok(None) => (Maps::default(), 0),
            Err(e) => {
                rejected_snapshot = Some(e.to_string());
                (Maps::default(), 0)
            }
        };

        if let Some(reason) = rejected_snapshot {
            // Delete it and say so in the manifest. Leaving it in place would let the next open
            // trust it again once the segment has grown back past the locations it names.
            warn!(
                "discarding the meta snapshot for {} ({}); replaying the log in full",
                dir.display(),
                reason
            );
            meta_snapshot::remove(&dir)?;
            manifest.snapshot_seq = 0;
            manifest.store(&dir)?;
        }

        let mut max_seq = snapshot_seq;
        for info in &manifest.sealed {
            if info.last_seq <= snapshot_seq {
                continue;
            }
            let segment = sealed
                .get(&info.id)
                .expect("sealed segment was just opened");
            let scan = scan_segment(segment.bytes(), info.id, snapshot_seq, &mut maps, false)?;
            max_seq = max_seq.max(scan.last_seq);
            debug!(
                "segment {}: applied {} records, skipped {}",
                info.id, scan.applied, scan.skipped
            );
        }

        let active_bytes = active.read_all()?;
        let scan = scan_segment(&active_bytes, active.id, snapshot_seq, &mut maps, true)?;
        max_seq = max_seq.max(scan.last_seq);
        active.first_seq = scan.first_seq;
        active.last_seq = scan.last_seq;
        if let Some(torn_at) = scan.torn_at {
            warn!(
                "truncating a torn tail in segment {} at offset {} ({} bytes discarded)",
                active.id,
                torn_at,
                active.len - torn_at
            );
            active.truncate_to(torn_at)?;
        }

        let mut store = Store {
            dir,
            opts,
            manifest,
            sealed,
            active,
            maps,
            pending_tombstone_bytes: 0,
        };
        store.manifest.next_seq = store.manifest.next_seq.max(max_seq + 1);
        info!(
            "opened shard {}: {} documents, {} splits, {} summaries, next seq {}",
            store.dir.display(),
            store.maps.docs.len(),
            store.maps.splits.len(),
            store.maps.summaries.len(),
            store.manifest.next_seq
        );
        Ok(store)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn options(&self) -> &StoreOptions {
        &self.opts
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn maps(&self) -> &Maps {
        &self.maps
    }

    pub fn doc_count(&self) -> usize {
        self.maps.docs.len()
    }

    pub fn split_count(&self) -> usize {
        self.maps.splits.len()
    }

    pub fn summary_count(&self) -> usize {
        self.maps.summaries.len()
    }

    pub fn contains_doc(&self, doc_id: u64) -> bool {
        self.maps.docs.contains_key(&doc_id)
    }

    // -----------------------------------------------------------------------
    // Writing
    // -----------------------------------------------------------------------

    /// Store a document, replacing any document with the same id.
    ///
    /// Nothing is written until every split and summary has been checked, so a rejected document
    /// leaves no partial state. A replacement writes a `DeleteDoc` first, which is what makes
    /// re-inserting a url whose content changed drop the old splits instead of orphaning them.
    ///
    /// The append phase is all-or-nothing: nothing touches the maps until every record is on the
    /// log, and a failure part way rewinds the segment, so the maps can never hold entities that
    /// a replay would not reproduce.
    pub fn insert(&mut self, doc: &DocumentDto) -> Result<Replaced> {
        let summaries = self.validate(doc)?;

        let rewind_len = self.active.len;
        let rewind_seq = self.manifest.next_seq;
        let appended = match self.append_document(doc, &summaries) {
            Ok(appended) => appended,
            Err(e) => {
                // Put the segment back where it was; the partially written records are unreachable
                // and would otherwise be replayed as a torn insert on the next open.
                if let Err(rewind_err) = self.active.truncate_to(rewind_len) {
                    return Err(e.context(format!(
                        "and the segment could not be rewound to {}: {}",
                        rewind_len, rewind_err
                    )));
                }
                self.manifest.next_seq = rewind_seq;
                return Err(e);
            }
        };

        let doc_id = doc.document_id;
        let mut replaced = Replaced::default();
        if appended.replaces {
            let active_id = self.active.id;
            let removed = self
                .maps
                .remove_doc(doc_id, active_id)
                .expect("document was present when the insert started");
            self.manifest.tombstone_bytes += removed.sealed_bytes;
            self.pending_tombstone_bytes += removed.active_bytes;
            replaced.existed = true;
            replaced.split_ids = removed.split_ids;
            replaced.summary_ids = removed.summary_ids;
        }

        for (summary_id, doc_id, split_id, loc) in appended.summaries {
            self.maps.summaries.insert(
                summary_id,
                SummaryEntry {
                    doc_id,
                    split_id,
                    loc,
                },
            );
        }
        for (split_id, doc_id, loc) in appended.splits {
            self.maps
                .splits
                .insert(split_id, SplitEntry { doc_id, loc });
        }
        self.maps.docs.insert(doc_id, appended.doc_entry);

        // Ids the new document reuses were overwritten in place, not removed: reporting them as
        // removed would make a caller drop vectors this very insert has just written.
        let live_splits: HashSet<u64> = doc.splits.iter().map(|s| s.split_id).collect();
        let live_summaries: HashSet<u64> = summaries.iter().map(|s| s.summary_id).collect();
        replaced.split_ids.retain(|id| !live_splits.contains(id));
        replaced
            .summary_ids
            .retain(|id| !live_summaries.contains(id));

        self.active.flush()?;
        self.seal_if_needed()?;
        Ok(replaced)
    }

    /// Write every record of a document to the log without touching the maps.
    fn append_document(&mut self, doc: &DocumentDto, summaries: &[SummaryDto]) -> Result<Appended> {
        let doc_id = doc.document_id;
        let replaces = self.maps.docs.contains_key(&doc_id);
        if replaces {
            let seq = self.next_seq();
            let meta = rmp_serde::to_vec_named(&DeleteMeta { doc_id })?;
            let bytes = encode(seq, RecordKind::DeleteDoc, VectorDtype::None, &meta, &[]);
            self.active.append(&bytes, seq)?;
        }

        // Summaries first, then splits, then the document: a torn insert replays as nothing.
        let mut summary_locs = Vec::with_capacity(summaries.len());
        for summary in summaries {
            let meta = rmp_serde::to_vec_named(&SummaryMeta {
                summary_id: summary.summary_id,
                doc_id: summary.document_id,
                split_id: summary.split_id,
                split_seq_id: summary.split_sequence_id,
                token_len: summary.token_len,
                centrality: summary.centrality,
                text: summary.text_content.clone(),
                model_id: summary
                    .embedding
                    .as_ref()
                    .map(|e| e.model_id)
                    .unwrap_or(self.opts.model_id),
            })?;
            let vector = encode_vector(
                &summary
                    .embedding
                    .as_ref()
                    .expect("checked by validate")
                    .embedding,
                self.opts.dtype,
            );
            let loc = self.append(RecordKind::Summary, &meta, &vector)?;
            summary_locs.push((
                summary.summary_id,
                summary.document_id,
                summary.split_id,
                loc,
            ));
        }

        let mut split_locs = Vec::with_capacity(doc.splits.len());
        for split in &doc.splits {
            let meta = rmp_serde::to_vec_named(&SplitMeta {
                split_id: split.split_id,
                seq_id: split.sequence_id,
                doc_id: split.doc_id,
                token_len: split.token_len,
                text: split.text_content.clone(),
                summary_ids: split.summaries.iter().map(|s| s.summary_id).collect(),
                model_id: split
                    .embedding
                    .as_ref()
                    .map(|e| e.model_id)
                    .unwrap_or(self.opts.model_id),
            })?;
            let vector = encode_vector(
                &split
                    .embedding
                    .as_ref()
                    .expect("checked by validate")
                    .embedding,
                self.opts.dtype,
            );
            let loc = self.append(RecordKind::Split, &meta, &vector)?;
            split_locs.push((split.split_id, split.doc_id, loc));
        }

        let doc_summary_ids: Option<Vec<u64>> = doc
            .summaries
            .as_ref()
            .map(|list| list.iter().map(|s| s.summary_id).collect());
        let meta = rmp_serde::to_vec_named(&DocMeta {
            doc_id,
            url: doc.document_url.clone(),
            split_ids: doc.splits.iter().map(|s| s.split_id).collect(),
            summary_ids: doc_summary_ids.clone(),
        })?;
        let seq = self.manifest.next_seq;
        let loc = self.append(RecordKind::Doc, &meta, &[])?;

        let in_doc_list: HashSet<u64> = doc_summary_ids.iter().flatten().copied().collect();
        let mut seen = HashSet::new();
        let extra_summary_ids: Vec<u64> = doc
            .splits
            .iter()
            .flat_map(|s| s.summaries.iter().map(|x| x.summary_id))
            .filter(|id| !in_doc_list.contains(id) && seen.insert(*id))
            .collect();

        Ok(Appended {
            replaces,
            summaries: summary_locs,
            splits: split_locs,
            doc_entry: DocEntry {
                url: doc.document_url.clone(),
                split_ids: doc.splits.iter().map(|s| s.split_id).collect(),
                summary_ids: doc_summary_ids,
                extra_summary_ids,
                seq,
                loc,
            },
        })
    }

    /// Remove a document and everything it owns.
    pub fn delete(&mut self, doc_id: u64) -> Result<Option<RemovedDoc>> {
        if !self.maps.docs.contains_key(&doc_id) {
            return Ok(None);
        }
        let seq = self.next_seq();
        let meta = rmp_serde::to_vec_named(&DeleteMeta { doc_id })?;
        let bytes = encode(seq, RecordKind::DeleteDoc, VectorDtype::None, &meta, &[]);
        self.active.append(&bytes, seq)?;
        let active_id = self.active.id;
        let removed = self
            .maps
            .remove_doc(doc_id, active_id)
            .expect("document was present a moment ago");
        self.manifest.tombstone_bytes += removed.sealed_bytes;
        self.pending_tombstone_bytes += removed.active_bytes;
        self.active.flush()?;
        self.seal_if_needed()?;
        Ok(Some(removed))
    }

    /// Every split and summary must carry an embedding of the right width from the right model,
    /// checked before anything is written. Returns the distinct summaries to store, the document's
    /// own list first and each id exactly once however many lists name it.
    fn validate(&self, doc: &DocumentDto) -> Result<Vec<SummaryDto>> {
        let mut distinct: Vec<SummaryDto> = Vec::new();
        let mut seen = HashSet::new();

        let check_embedding =
            |what: &str, id: u64, embedding: &Option<EmbeddingDto>| -> Result<()> {
                let Some(e) = embedding else {
                    return Err(anyhow!("{} {} has no embedding", what, id));
                };
                if e.embedding.len() != self.opts.n_embd {
                    return Err(anyhow!(
                        "{} {} has {} dimensions, shard holds {}",
                        what,
                        id,
                        e.embedding.len(),
                        self.opts.n_embd
                    ));
                }
                if e.model_id != self.opts.model_id {
                    return Err(anyhow!(
                        "{} {} comes from model {:#x}, shard holds {:#x}",
                        what,
                        id,
                        e.model_id,
                        self.opts.model_id
                    ));
                }
                Ok(())
            };

        for summary in doc.summaries.iter().flatten() {
            check_embedding("summary", summary.summary_id, &summary.embedding)?;
            if seen.insert(summary.summary_id) {
                distinct.push(summary.clone());
            }
        }
        for split in &doc.splits {
            check_embedding("split", split.split_id, &split.embedding)?;
            for summary in &split.summaries {
                check_embedding("summary", summary.summary_id, &summary.embedding)?;
                if seen.insert(summary.summary_id) {
                    distinct.push(summary.clone());
                }
            }
        }
        Ok(distinct)
    }

    fn next_seq(&mut self) -> u64 {
        let seq = self.manifest.next_seq;
        self.manifest.next_seq += 1;
        seq
    }

    fn append(&mut self, kind: RecordKind, meta: &[u8], vector: &[u8]) -> Result<Loc> {
        let dtype = if vector.is_empty() {
            VectorDtype::None
        } else {
            self.opts.dtype
        };
        let seq = self.next_seq();
        let bytes = encode(seq, kind, dtype, meta, vector);
        let offset = self.active.append(&bytes, seq)?;
        Ok(Loc {
            segment: self.active.id,
            offset,
            len: bytes.len() as u32,
        })
    }

    /// Seal the active segment when it has grown past the configured size. Only ever called at an
    /// insert boundary, which is what keeps all of a document's records inside one segment.
    fn seal_if_needed(&mut self) -> Result<()> {
        if self.active.len < self.opts.segment_max_bytes || self.active.len == 0 {
            return Ok(());
        }
        self.seal_active()
    }

    pub fn seal_active(&mut self) -> Result<()> {
        if self.active.len == 0 {
            return Ok(());
        }
        let new_id = self.manifest.next_segment_id;
        self.manifest.next_segment_id += 1;
        let new_active = ActiveSegment::open(&self.dir, new_id)?;
        let old = std::mem::replace(&mut self.active, new_active);

        let sealed_id = old.id;
        let info = SegmentInfo {
            id: sealed_id,
            bytes: old.len,
            first_seq: old.first_seq.unwrap_or(0),
            last_seq: old.last_seq,
        };
        let sealed = old.seal()?;
        self.sealed.insert(info.id, sealed);
        self.manifest.sealed.push(info);
        self.manifest.active = new_id;
        // Records that died while this segment was active are now dead bytes in a sealed segment,
        // which is exactly what the compaction ratio measures.
        self.manifest.tombstone_bytes += self.pending_tombstone_bytes;
        self.pending_tombstone_bytes = 0;
        self.manifest.store(&self.dir)?;
        info!("sealed segment {}, new active {}", sealed_id, new_id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Durability
    // -----------------------------------------------------------------------

    /// Put every acknowledged write on the device.
    pub fn fsync(&mut self) -> Result<()> {
        self.active.sync()
    }

    /// Write `meta.snap` and record in the manifest how far it is caught up.
    ///
    /// The log is fsynced first. Publishing a `snapshot_seq` that covers records still sitting in
    /// the page cache would, after a power loss, leave the maps pointing into a segment that is
    /// shorter than they believe: replay would skip those records as already covered, find no torn
    /// tail to truncate, and the next append would land on top of locations the snapshot still
    /// names.
    pub fn snapshot_meta(&mut self) -> Result<u64> {
        self.active.sync()?;
        let snapshot_seq = self.manifest.next_seq.saturating_sub(1);
        meta_snapshot::store(&self.dir, &self.maps, snapshot_seq)?;
        self.manifest.snapshot_seq = snapshot_seq;
        self.manifest.store(&self.dir)?;
        Ok(snapshot_seq)
    }

    // -----------------------------------------------------------------------
    // Reading
    // -----------------------------------------------------------------------

    /// Hand the bytes at `loc` to `f`, without copying them when the segment is sealed.
    ///
    /// A sealed segment is an immutable mapping, so its bytes are borrowed straight from it; the
    /// active segment is a file being appended to and has to be read into a buffer first.
    pub(crate) fn with_bytes<R>(&self, loc: Loc, f: impl FnOnce(&[u8]) -> R) -> Result<R> {
        if loc.segment == self.active.id {
            let buf = self.active.read_at(loc.offset, loc.len)?;
            Ok(f(&buf))
        } else {
            let segment = self
                .sealed
                .get(&loc.segment)
                .ok_or_else(|| anyhow!("segment {} is not open", loc.segment))?;
            Ok(f(segment.slice(loc.offset, loc.len)?))
        }
    }

    /// Hand one entity's stored vector to `f` as the bytes in the log, with their dtype.
    ///
    /// Every record of a searchable kind ends with its vector, and every vector in a shard is the
    /// same width, so this reads `n_embd * bytes_per_element` bytes off the end of the record
    /// rather than the whole record: the text is the bulk of a record and a distance computation
    /// has no use for it. The frame was checksummed when it was written and again on replay, so
    /// the CRC is not re-checked here.
    pub fn with_vector<R>(
        &self,
        kind: EntityKind,
        id: u64,
        f: impl FnOnce(&[u8], VectorDtype) -> R,
    ) -> Result<Option<R>> {
        let loc = match kind {
            EntityKind::Split => self.maps.splits.get(&id).map(|e| e.loc),
            EntityKind::Summary => self.maps.summaries.get(&id).map(|e| e.loc),
        };
        let Some(loc) = loc else {
            return Ok(None);
        };
        let vector_len = self.opts.dtype.vector_bytes(self.opts.n_embd) as u32;
        if loc.len < crate::record::RECORD_HEADER_LEN as u32 + vector_len {
            return Err(anyhow!(
                "record for {} is {} bytes, too short to hold a {} byte vector",
                id,
                loc.len,
                vector_len
            ));
        }
        let tail = Loc {
            segment: loc.segment,
            offset: loc.offset + (loc.len - vector_len) as u64,
            len: vector_len,
        };
        let dtype = self.opts.dtype;
        self.with_bytes(tail, |bytes| f(bytes, dtype)).map(Some)
    }

    pub(crate) fn read_record(&self, loc: Loc) -> Result<Vec<u8>> {
        if loc.segment == self.active.id {
            self.active.read_at(loc.offset, loc.len)
        } else {
            let segment = self
                .sealed
                .get(&loc.segment)
                .ok_or_else(|| anyhow!("segment {} is not open", loc.segment))?;
            Ok(segment.slice(loc.offset, loc.len)?.to_vec())
        }
    }

    fn read_split(
        &self,
        split_id: u64,
        with_embedding: bool,
    ) -> Result<Option<(SplitMeta, Option<EmbeddingDto>)>> {
        let Some(entry) = self.maps.splits.get(&split_id) else {
            return Ok(None);
        };
        let bytes = self.read_record(entry.loc)?;
        let view = decode(&bytes).map_err(|e| anyhow!("split {}: {}", split_id, e))?;
        let meta: SplitMeta = rmp_serde::from_slice(view.meta)?;
        let embedding = if with_embedding {
            Some(EmbeddingDto::new(
                split_id,
                decode_vector(view.vector, view.dtype),
                meta.model_id,
            ))
        } else {
            None
        };
        Ok(Some((meta, embedding)))
    }

    fn read_summary(
        &self,
        summary_id: u64,
        with_embedding: bool,
    ) -> Result<Option<(SummaryMeta, Option<EmbeddingDto>)>> {
        let Some(entry) = self.maps.summaries.get(&summary_id) else {
            return Ok(None);
        };
        let bytes = self.read_record(entry.loc)?;
        let view = decode(&bytes).map_err(|e| anyhow!("summary {}: {}", summary_id, e))?;
        let meta: SummaryMeta = rmp_serde::from_slice(view.meta)?;
        let embedding = if with_embedding {
            Some(EmbeddingDto::new(
                summary_id,
                decode_vector(view.vector, view.dtype),
                meta.model_id,
            ))
        } else {
            None
        };
        Ok(Some((meta, embedding)))
    }

    /// One summary, or `None` when the id is unknown.
    pub fn get_summary(
        &self,
        summary_id: u64,
        with_embeddings: bool,
    ) -> Result<Option<SummaryDto>> {
        Ok(self
            .read_summary(summary_id, with_embeddings)?
            .map(|(meta, embedding)| summary_dto(meta, embedding)))
    }

    /// Summaries in the order asked for, skipping ids that are not in this shard.
    pub fn get_summaries(&self, ids: &[u64], with_embeddings: bool) -> Result<Vec<SummaryDto>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(dto) = self.get_summary(*id, with_embeddings)? {
                out.push(dto);
            }
        }
        Ok(out)
    }

    /// One split with its own summary list, in recorded order.
    pub fn get_split(&self, split_id: u64, with_embeddings: bool) -> Result<Option<SplitDto>> {
        let Some((meta, embedding)) = self.read_split(split_id, with_embeddings)? else {
            return Ok(None);
        };
        let summaries = self.get_summaries(&meta.summary_ids, with_embeddings)?;
        Ok(Some(split_dto(meta, summaries, embedding, None)))
    }

    /// One split with an empty summary list.
    ///
    /// The query path attaches the summaries the query actually hit rather than the split's whole
    /// list, so reading that list would be work thrown away.
    pub fn get_split_without_summaries(
        &self,
        split_id: u64,
        with_embeddings: bool,
    ) -> Result<Option<SplitDto>> {
        Ok(self
            .read_split(split_id, with_embeddings)?
            .map(|(meta, embedding)| split_dto(meta, Vec::new(), embedding, None)))
    }

    /// Splits in the order asked for, each with its own summary list.
    pub fn get_splits(&self, ids: &[u64], with_embeddings: bool) -> Result<Vec<SplitDto>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(dto) = self.get_split(*id, with_embeddings)? {
                out.push(dto);
            }
        }
        Ok(out)
    }

    /// A document with every split, every split's summaries and the document's own summary list,
    /// all in recorded order. This is what document retrieval by id or url returns.
    pub fn get_doc(&self, doc_id: u64, with_embeddings: bool) -> Result<Option<DocumentDto>> {
        let Some(entry) = self.maps.docs.get(&doc_id) else {
            return Ok(None);
        };
        let splits = self.get_splits(&entry.split_ids, with_embeddings)?;
        let summaries = match &entry.summary_ids {
            Some(ids) => Some(self.get_summaries(ids, with_embeddings)?),
            None => None,
        };
        Ok(Some(DocumentDto::new(
            doc_id, &entry.url, splits, summaries,
        )))
    }

    /// The document id a url maps to in this shard, if it is present.
    ///
    /// This is a linear scan and is meant for diagnostics and for the migration tool. The server
    /// path never needs it: `doc_id` is a deterministic hash of the url, so a caller that has the
    /// same hasher looks the document up by id directly.
    pub fn doc_id_for_url(&self, url: &str) -> Option<u64> {
        self.maps
            .docs
            .iter()
            .find(|(_, e)| e.url == url)
            .map(|(id, _)| *id)
    }

    /// Hand every live vector of one kind to `f`, as the bytes stored in the log together with
    /// their dtype. Used to rebuild a usearch index without going through `f32`.
    pub fn for_each_live_vector<F>(&self, kind: EntityKind, mut f: F) -> Result<()>
    where
        F: FnMut(u64, &[u8], VectorDtype) -> Result<()>,
    {
        let locs: Vec<(u64, Loc)> = match kind {
            EntityKind::Split => self
                .maps
                .splits
                .iter()
                .map(|(id, e)| (*id, e.loc))
                .collect(),
            EntityKind::Summary => self
                .maps
                .summaries
                .iter()
                .map(|(id, e)| (*id, e.loc))
                .collect(),
        };
        for (id, loc) in locs {
            let bytes = self.read_record(loc)?;
            let view = decode(&bytes).map_err(|e| anyhow!("record {}: {}", id, e))?;
            f(id, view.vector, view.dtype)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Compaction
    // -----------------------------------------------------------------------

    /// Whether dead records take up more than `ratio` of the sealed segments.
    ///
    /// Separate from the rewrite so a caller can find out before doing anything expensive: a
    /// shard whose indexes are only mapped has to read them in first, and that is not worth doing
    /// unless there is really something to compact.
    pub fn needs_compaction(&self, ratio: f64) -> bool {
        let sealed_bytes = self.manifest.sealed_bytes();
        sealed_bytes > 0 && (self.manifest.tombstone_bytes as f64) / (sealed_bytes as f64) > ratio
    }

    /// Rewrite the sealed segments when dead records take up more than `ratio` of them.
    pub fn compact_if_needed(&mut self, ratio: f64) -> Result<bool> {
        if !self.needs_compaction(ratio) {
            return Ok(false);
        }
        self.compact()?;
        Ok(true)
    }

    pub fn compact(&mut self) -> Result<()> {
        crate::compact::compact(self)
    }
}

/// Does every location in the maps sit inside a segment that is actually that long?
fn locations_fit(
    maps: &Maps,
    sealed: &BTreeMap<u32, SealedSegment>,
    active: &ActiveSegment,
) -> bool {
    let fits = |loc: &Loc| -> bool {
        let end = loc.offset + loc.len as u64;
        if loc.segment == active.id {
            end <= active.len
        } else {
            match sealed.get(&loc.segment) {
                Some(segment) => end <= segment.len(),
                None => false,
            }
        }
    };
    maps.docs.values().all(|e| fits(&e.loc))
        && maps.splits.values().all(|e| fits(&e.loc))
        && maps.summaries.values().all(|e| fits(&e.loc))
}

/// Delete `records-*.seg` files the manifest does not reference. A crash during compaction, after
/// the replacement segment is written but before the manifest names it, leaves one behind.
fn remove_orphan_segments(dir: &Path, manifest: &Manifest) -> Result<()> {
    let mut known: HashSet<u32> = manifest.segment_ids().into_iter().collect();
    known.insert(manifest.active);
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix("records-") else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".seg") else {
            continue;
        };
        let Ok(id) = digits.parse::<u32>() else {
            continue;
        };
        if !known.contains(&id) {
            warn!("removing orphan segment {}", entry.path().display());
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DTO construction
// ---------------------------------------------------------------------------

pub(crate) fn summary_dto(meta: SummaryMeta, embedding: Option<EmbeddingDto>) -> SummaryDto {
    SummaryDto::new(
        meta.summary_id,
        meta.doc_id,
        meta.split_id,
        meta.split_seq_id,
        &meta.text,
        meta.token_len,
        meta.centrality,
        embedding,
        None,
        None,
    )
}

pub(crate) fn split_dto(
    meta: SplitMeta,
    summaries: Vec<SummaryDto>,
    embedding: Option<EmbeddingDto>,
    query_distance: Option<f32>,
) -> SplitDto {
    SplitDto::new(
        meta.split_id,
        meta.seq_id,
        meta.doc_id,
        &meta.text,
        meta.token_len,
        summaries,
        embedding,
        query_distance,
        None,
    )
}
