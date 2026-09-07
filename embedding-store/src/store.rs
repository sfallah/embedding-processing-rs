//! `Store`: the record log plus the maps, one per workspace shard.
//!
//! The store owns durability and layout. It knows nothing about vector search: the shard in step 3
//! wraps it together with the two usearch indexes and a lock.

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
    /// The model every embedding in this shard must come from (decision D5).
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
#[derive(Debug, Clone, Default)]
pub struct Replaced {
    pub existed: bool,
    pub split_ids: Vec<u64>,
    pub summary_ids: Vec<u64>,
}

pub struct Store {
    pub(crate) dir: PathBuf,
    pub(crate) opts: StoreOptions,
    pub(crate) manifest: Manifest,
    pub(crate) sealed: BTreeMap<u32, SealedSegment>,
    pub(crate) active: ActiveSegment,
    pub(crate) maps: Maps,
}

impl Store {
    // -----------------------------------------------------------------------
    // Opening
    // -----------------------------------------------------------------------

    pub fn open(dir: impl AsRef<Path>, opts: StoreOptions) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;

        let manifest = match Manifest::load(&dir)? {
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

        // The snapshot only counts if it is the one this manifest describes.
        let (mut maps, snapshot_seq) = match meta_snapshot::load(&dir) {
            Ok(Some((maps, seq))) if seq == manifest.snapshot_seq => (maps, seq),
            Ok(Some((_, seq))) => {
                warn!(
                    "ignoring meta snapshot at seq {} (manifest says {}); replaying the log",
                    seq, manifest.snapshot_seq
                );
                (Maps::default(), 0)
            }
            Ok(None) => (Maps::default(), 0),
            Err(e) => {
                warn!("meta snapshot unusable ({}); replaying the log", e);
                (Maps::default(), 0)
            }
        };

        let mut max_seq = snapshot_seq;
        for info in &manifest.sealed {
            if info.last_seq <= snapshot_seq {
                continue;
            }
            let segment = sealed.get(&info.id).expect("sealed segment was just opened");
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
    pub fn insert(&mut self, doc: &DocumentDto) -> Result<Replaced> {
        let summaries = self.validate(doc)?;

        let doc_id = doc.document_id;
        let mut replaced = Replaced::default();
        if self.maps.docs.contains_key(&doc_id) {
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
            replaced.existed = true;
            replaced.split_ids = removed.split_ids;
            replaced.summary_ids = removed.summary_ids;
        }

        // Summaries first, then splits, then the document: a torn insert replays as nothing.
        for summary in &summaries {
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
                &summary.embedding.as_ref().expect("checked above").embedding,
                self.opts.dtype,
            );
            let loc = self.append(RecordKind::Summary, &meta, &vector)?;
            self.maps.summaries.insert(
                summary.summary_id,
                SummaryEntry {
                    doc_id: summary.document_id,
                    split_id: summary.split_id,
                    loc,
                },
            );
        }

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
                &split.embedding.as_ref().expect("checked above").embedding,
                self.opts.dtype,
            );
            let loc = self.append(RecordKind::Split, &meta, &vector)?;
            self.maps.splits.insert(
                split.split_id,
                SplitEntry {
                    doc_id: split.doc_id,
                    loc,
                },
            );
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

        self.maps.docs.insert(
            doc_id,
            DocEntry {
                url: doc.document_url.clone(),
                split_ids: doc.splits.iter().map(|s| s.split_id).collect(),
                summary_ids: doc_summary_ids,
                extra_summary_ids,
                seq,
                loc,
            },
        );

        self.active.flush()?;
        self.seal_if_needed()?;
        Ok(replaced)
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

        let check_embedding = |what: &str, id: u64, embedding: &Option<EmbeddingDto>| -> Result<()> {
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

        let info = SegmentInfo {
            id: old.id,
            bytes: old.len,
            first_seq: old.first_seq.unwrap_or(0),
            last_seq: old.last_seq,
        };
        let sealed = old.seal()?;
        self.sealed.insert(info.id, sealed);
        self.manifest.sealed.push(info);
        self.manifest.active = new_id;
        self.manifest.store(&self.dir)?;
        info!("sealed segment {}, new active {}", new_id - 1, new_id);
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
    pub fn snapshot_meta(&mut self) -> Result<u64> {
        self.active.flush()?;
        let snapshot_seq = self.manifest.next_seq.saturating_sub(1);
        meta_snapshot::store(&self.dir, &self.maps, snapshot_seq)?;
        self.manifest.snapshot_seq = snapshot_seq;
        self.manifest.store(&self.dir)?;
        Ok(snapshot_seq)
    }

    // -----------------------------------------------------------------------
    // Reading
    // -----------------------------------------------------------------------

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

    fn read_split(&self, split_id: u64, with_embedding: bool) -> Result<Option<(SplitMeta, Option<EmbeddingDto>)>> {
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

    fn read_summary(&self, summary_id: u64, with_embedding: bool) -> Result<Option<(SummaryMeta, Option<EmbeddingDto>)>> {
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
    pub fn get_summary(&self, summary_id: u64, with_embeddings: bool) -> Result<Option<SummaryDto>> {
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
            doc_id,
            &entry.url,
            splits,
            summaries,
        )))
    }

    /// The document id a url maps to in this shard, if it is present.
    pub fn doc_id_for_url(&self, url: &str) -> Option<u64> {
        self.maps
            .docs
            .iter()
            .find(|(_, e)| e.url == url)
            .map(|(id, _)| *id)
    }

    /// Hand every live vector of one kind to `f`, as the bytes stored in the log. Used to rebuild
    /// a usearch index without going through `f32`.
    pub fn for_each_live_vector<F>(&self, kind: EntityKind, mut f: F) -> Result<()>
    where
        F: FnMut(u64, &[u8]) -> Result<()>,
    {
        let locs: Vec<(u64, Loc)> = match kind {
            EntityKind::Split => self.maps.splits.iter().map(|(id, e)| (*id, e.loc)).collect(),
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
            f(id, view.vector)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Compaction
    // -----------------------------------------------------------------------

    /// Rewrite the sealed segments when dead records take up more than `ratio` of them.
    pub fn compact_if_needed(&mut self, ratio: f64) -> Result<bool> {
        let sealed_bytes = self.manifest.sealed_bytes();
        if sealed_bytes == 0 {
            return Ok(false);
        }
        if (self.manifest.tombstone_bytes as f64) / (sealed_bytes as f64) <= ratio {
            return Ok(false);
        }
        self.compact()?;
        Ok(true)
    }

    pub fn compact(&mut self) -> Result<()> {
        crate::compact::compact(self)
    }

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
