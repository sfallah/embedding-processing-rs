//! Reads a store written by the previous storage layer and writes it into per-workspace shards.
//!
//! Behind the `rocksdb-migration` feature, which is now the only place the `rocksdb` crate
//! remains: `embedding-database` is deleted. It reads the old column families directly,
//! keeping its own copy of the two things about the old format it needs — the column family names
//! and the key encoding — which is what let it outlive that crate.
//!
//! The old layer kept the workspace in exactly one place, an `embedding_filter_info` row per
//! split and per summary vector holding `(embed_id, doc_id, user_uuid)`. That is what decides
//! which shard a document belongs in. A document named by two workspaces is written to both:
//! a shard *is* a workspace, and each has to answer for that document on its own. Two workspaces
//! can only claim one document if its own rows disagree, since a row is keyed by vector id and
//! the ids are derived from the url; that is what a half-overwritten document looks like, and
//! each workspace gets the whole document rather than the half its rows name, because a document
//! is what a shard stores.
//!
//! Documents that no filter row names are unreachable in the old layer too, since a search there
//! is filtered by user uuid, so they are counted and left behind rather than migrated.

use crate::pool::ShardPool;
use crate::shard::ShardOptions;
use anyhow::{anyhow, Context, Result};
use embedding_common::prelude::*;
use rocksdb::{IteratorMode, Options, DB};
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// The old layer's column families, from the deleted `embedding-database`'s `ColumnFamilyType`.
/// Spelled out rather than imported, which is what let this outlive that crate. All of them must
/// be named at open time, or RocksDB refuses the directory.
const CF_DEFAULT: &str = "default";
const CF_DOCUMENTS: &str = "documents";
const CF_SPLITS: &str = "splits";
const CF_SUMMARIES: &str = "summaries";
const CF_EMBEDDINGS: &str = "embeddings";
const CF_FILTER_INFO: &str = "embedding_filter_info";
const CF_MODELS: &str = "models";

pub const ALL_CFS: [&str; 7] = [
    CF_DEFAULT,
    CF_DOCUMENTS,
    CF_SPLITS,
    CF_SUMMARIES,
    CF_EMBEDDINGS,
    CF_FILTER_INFO,
    CF_MODELS,
];

/// The old layer wrote its u64 keys little-endian, through `byteorder::LittleEndian::write_u64`.
pub fn key(id: u64) -> [u8; 8] {
    id.to_le_bytes()
}

pub struct MigrationOptions {
    /// The RocksDB directory to read.
    pub source: PathBuf,
    /// Where the shards go.
    pub destination: PathBuf,
    /// What the destination shards are opened with. Its model id is what refuses vectors from
    /// another model.
    pub shard: ShardOptions,
    pub memory_budget_mb: usize,
    /// Read and count everything, write nothing.
    pub dry_run: bool,
    /// Open the source read-write. The default open is read-only and leaves the source
    /// untouched; RocksDB replays the write-ahead log either way, which matters because a
    /// database that was never flushed keeps everything there and has no SST files at all.
    pub read_write: bool,
}

#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counts {
    pub docs: usize,
    pub splits: usize,
    pub summaries: usize,
}

impl Counts {
    fn add(&mut self, other: Counts) {
        self.docs += other.docs;
        self.splits += other.splits;
        self.summaries += other.summaries;
    }
}

#[derive(Debug)]
pub struct WorkspaceReport {
    pub workspace: Uuid,
    /// What was read out of the source and accepted by the shard.
    pub read: Counts,
    /// What the shard holds afterwards. `None` on a dry run.
    pub written: Option<Counts>,
}

#[derive(Debug, Default)]
pub struct MigrationReport {
    pub filter_rows: usize,
    pub distinct_documents: usize,
    /// Documents belonging to more than one workspace. Each is written to every one of them.
    pub shared_documents: usize,
    /// Documents in the source that no filter row names.
    pub orphan_documents: usize,
    pub workspaces: Vec<WorkspaceReport>,
    pub totals: Counts,
    pub shards_snapshotted: usize,
    /// One line per document that could not be migrated. The migration continues past a failure,
    /// so that one run shows the whole picture rather than one broken document at a time.
    pub failures: Vec<String>,
}

pub fn migrate(opts: &MigrationOptions) -> Result<MigrationReport> {
    let db = open_source(&opts.source, opts.read_write)?;
    let by_workspace = scan_workspaces(&db)?;

    let mut report = MigrationReport {
        filter_rows: count_filter_rows(&db)?,
        distinct_documents: distinct_documents(&by_workspace),
        shared_documents: shared_documents(&by_workspace),
        orphan_documents: orphan_documents(&db, &by_workspace)?,
        ..Default::default()
    };

    let pool = if opts.dry_run {
        None
    } else {
        Some(
            ShardPool::new(&opts.destination, opts.shard.clone())?
                .with_memory_budget_mb(opts.memory_budget_mb),
        )
    };

    for (workspace, doc_ids) in &by_workspace {
        let mut read = Counts::default();
        let shard = match pool.as_ref() {
            Some(pool) => Some(pool.get(*workspace)?),
            None => None,
        };

        for doc_id in doc_ids {
            let dto = match read_document(&db, *doc_id) {
                Ok(Some(dto)) => dto,
                Ok(None) => {
                    report.failures.push(format!(
                        "{workspace} document {doc_id}: named by a filter row but not in the documents column family"
                    ));
                    continue;
                }
                Err(e) => {
                    report
                        .failures
                        .push(format!("{workspace} document {doc_id}: {e:#}"));
                    continue;
                }
            };

            if let Some(shard) = shard.as_ref() {
                if let Err(e) = shard.insert(&dto) {
                    report
                        .failures
                        .push(format!("{workspace} document {doc_id}: {e:#}"));
                    continue;
                }
            }

            read.docs += 1;
            read.splits += dto.splits.len();
            read.summaries += distinct_summaries(&dto);
        }

        let written = shard.as_ref().map(|shard| {
            let stats = shard.stats();
            Counts {
                docs: stats.docs,
                splits: stats.splits,
                summaries: stats.summaries,
            }
        });
        report.totals.add(read);
        report.workspaces.push(WorkspaceReport {
            workspace: *workspace,
            read,
            written,
        });
    }

    if let Some(pool) = pool.as_ref() {
        report.shards_snapshotted = pool.snapshot_all()?;
    }

    Ok(report)
}

fn open_source(path: &Path, read_write: bool) -> Result<DB> {
    if !path.exists() {
        return Err(anyhow!("{} does not exist", path.display()));
    }
    let mut opts = Options::default();
    opts.create_if_missing(false);

    let db = if read_write {
        DB::open_cf(&opts, path, ALL_CFS)
    } else {
        DB::open_cf_for_read_only(&opts, path, ALL_CFS, false)
    }
    .with_context(|| format!("opening {} as a RocksDB store", path.display()))?;
    Ok(db)
}

/// Groups documents by workspace, from the filter rows. There is one row per vector, so a
/// document with six splits and twelve summaries contributes eighteen rows naming it.
fn scan_workspaces(db: &DB) -> Result<BTreeMap<Uuid, BTreeSet<u64>>> {
    let cf = db
        .cf_handle(CF_FILTER_INFO)
        .ok_or_else(|| anyhow!("the source has no {CF_FILTER_INFO} column family"))?;

    let mut by_workspace: BTreeMap<Uuid, BTreeSet<u64>> = BTreeMap::new();
    for item in db.iterator_cf(&cf, IteratorMode::Start) {
        let (_key, value) = item?;
        let info = EmbeddingFilterInfo::unpack::<EmbeddingFilterInfo>(&value)
            .map_err(|e| anyhow!("unpacking a filter row: {e}"))?;
        by_workspace
            .entry(info.user_uuid)
            .or_default()
            .insert(info.doc_id);
    }
    Ok(by_workspace)
}

fn count_filter_rows(db: &DB) -> Result<usize> {
    let cf = db
        .cf_handle(CF_FILTER_INFO)
        .ok_or_else(|| anyhow!("the source has no {CF_FILTER_INFO} column family"))?;
    Ok(db.iterator_cf(&cf, IteratorMode::Start).count())
}

fn distinct_documents(by_workspace: &BTreeMap<Uuid, BTreeSet<u64>>) -> usize {
    by_workspace
        .values()
        .flat_map(|docs| docs.iter().copied())
        .collect::<HashSet<u64>>()
        .len()
}

fn shared_documents(by_workspace: &BTreeMap<Uuid, BTreeSet<u64>>) -> usize {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut shared: HashSet<u64> = HashSet::new();
    for docs in by_workspace.values() {
        for doc_id in docs {
            if !seen.insert(*doc_id) {
                shared.insert(*doc_id);
            }
        }
    }
    shared.len()
}

fn orphan_documents(db: &DB, by_workspace: &BTreeMap<Uuid, BTreeSet<u64>>) -> Result<usize> {
    let named: HashSet<u64> = by_workspace
        .values()
        .flat_map(|docs| docs.iter().copied())
        .collect();
    let cf = db
        .cf_handle(CF_DOCUMENTS)
        .ok_or_else(|| anyhow!("the source has no {CF_DOCUMENTS} column family"))?;

    let mut orphans = 0usize;
    for item in db.iterator_cf(&cf, IteratorMode::Start) {
        let (key, _value) = item?;
        let bytes: [u8; 8] = key
            .as_ref()
            .try_into()
            .map_err(|_| anyhow!("a document key is not eight bytes"))?;
        if !named.contains(&u64::from_le_bytes(bytes)) {
            orphans += 1;
        }
    }
    Ok(orphans)
}

fn get<T>(db: &DB, cf: &str, id: u64) -> Result<Option<T>>
where
    T: Serde + DeserializeOwned,
{
    let handle = db
        .cf_handle(cf)
        .ok_or_else(|| anyhow!("the source has no {cf} column family"))?;
    match db.get_cf(&handle, key(id))? {
        Some(bytes) => {
            let value =
                T::unpack::<T>(&bytes).map_err(|e| anyhow!("unpacking {cf} record {id}: {e}"))?;
            Ok(Some(value))
        }
        None => Ok(None),
    }
}

/// Joins a document with its splits, their summaries and every vector, into the same DTO the
/// server would have inserted.
fn read_document(db: &DB, doc_id: u64) -> Result<Option<DocumentDto>> {
    let Some(doc) = get::<Document>(db, CF_DOCUMENTS, doc_id)? else {
        return Ok(None);
    };

    let mut split_dtos = Vec::with_capacity(doc.split_ids.len());
    for split_id in &doc.split_ids {
        let split = get::<Split>(db, CF_SPLITS, *split_id)?.ok_or_else(|| {
            anyhow!("split {split_id} is named by the document but is not stored")
        })?;
        let mut summaries = Vec::new();
        for summary_id in split.summary_ids.clone().unwrap_or_default() {
            summaries.push(read_summary(db, summary_id)?);
        }
        let embedding = read_embedding(db, *split_id, "split")?;
        split_dtos.push(split.to_dto_full(summaries, Some(embedding), None));
    }

    // Document-level summaries are a separate list in the old model. The store takes the union of
    // the two, so both are handed over as they are.
    let doc_summaries = match doc.summary_ids.as_ref() {
        Some(ids) if !ids.is_empty() => {
            let mut summaries = Vec::with_capacity(ids.len());
            for summary_id in ids {
                summaries.push(read_summary(db, *summary_id)?);
            }
            Some(summaries)
        }
        _ => None,
    };

    Ok(Some(doc.to_dto(&split_dtos, doc_summaries)))
}

fn read_summary(db: &DB, summary_id: u64) -> Result<SummaryDto> {
    let summary = get::<Summary>(db, CF_SUMMARIES, summary_id)?
        .ok_or_else(|| anyhow!("summary {summary_id} is named but is not stored"))?;
    let embedding = read_embedding(db, summary_id, "summary")?;
    Ok(summary.to_dto(Some(embedding), None))
}

fn read_embedding(db: &DB, id: u64, what: &str) -> Result<EmbeddingDto> {
    let embedding = get::<Embedding>(db, CF_EMBEDDINGS, id)?
        .ok_or_else(|| anyhow!("{what} {id} has no embedding"))?;
    Ok(embedding.to_dto())
}

/// A summary can be named by the document and by its split. It is one entity with one vector, so
/// it is counted once, the way the store stores it.
fn distinct_summaries(doc: &DocumentDto) -> usize {
    let mut seen = HashSet::new();
    for summary in doc.summaries.iter().flatten() {
        seen.insert(summary.summary_id);
    }
    for split in &doc.splits {
        for summary in &split.summaries {
            seen.insert(summary.summary_id);
        }
    }
    seen.len()
}
