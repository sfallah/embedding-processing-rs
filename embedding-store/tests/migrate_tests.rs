//! Step 5 tests: reading a store written by the previous storage layer.
//!
//! These build their own RocksDB in the old shape rather than leaning on the `rocksdb_dir/` in
//! the repo root, which is not checked in. What is worth testing is the part that is not a
//! straight copy: the workspace lives only in the filter rows, one per vector, so grouping,
//! documents belonging to two workspaces, and documents belonging to none are all decided there.

#![cfg(feature = "rocksdb-migration")]

use embedding_common::config::{IndexConfig, MetricKind, ScalarKind};
use embedding_common::prelude::*;
use embedding_store::migrate::{key, migrate, MigrationOptions, ALL_CFS};
use embedding_store::prelude::{ShardOptions, ShardPool};
use rocksdb::{Options, DB};
use serde::Serialize;
use std::path::Path;
use tempfile::TempDir;
use uuid::Uuid;

const N_EMBD: usize = 16;
const MODEL_ID: u64 = 0xA11CE;
const OTHER_MODEL_ID: u64 = 0xB0B;

const CF_DOCUMENTS: &str = "documents";
const CF_SPLITS: &str = "splits";
const CF_SUMMARIES: &str = "summaries";
const CF_EMBEDDINGS: &str = "embeddings";
const CF_FILTER_INFO: &str = "embedding_filter_info";

fn index_config() -> IndexConfig {
    IndexConfig {
        index_dir: "unused".to_string(),
        dimensions: N_EMBD,
        metric_kind: MetricKind::Cos,
        scalar_kind: ScalarKind::F16,
        connectivity: 16,
        expansion_add: 200,
        expansion_search: 32,
    }
}

fn vector(salt: u64) -> Vec<f32> {
    let mut state = salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut raw: Vec<f32> = (0..N_EMBD)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        })
        .collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in raw.iter_mut() {
        *x /= norm;
    }
    raw
}

/// A store in the old shape: five column families keyed by little-endian u64, a vector per split
/// and per summary, and a filter row per vector carrying the workspace.
struct Source {
    db: DB,
}

impl Source {
    fn create(path: &Path) -> Self {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        Source {
            db: DB::open_cf(&opts, path, ALL_CFS).unwrap(),
        }
    }

    fn put<T: Serde + Serialize>(&self, cf: &str, id: u64, value: &T) {
        let handle = self.db.cf_handle(cf).unwrap();
        self.db
            .put_cf(&handle, key(id), value.pack().unwrap())
            .unwrap();
    }

    /// Writes a document, its splits, one summary per split, and the vectors, exactly as the old
    /// layer's `save_doc` did. Returns every vector id, which is what a filter row is keyed by.
    fn write_document(
        &self,
        doc_id: u64,
        n_splits: usize,
        model_id: u64,
        with_embeddings: bool,
    ) -> Vec<u64> {
        let mut split_ids = Vec::new();
        let mut summary_ids = Vec::new();

        for seq in 0..n_splits {
            let split_id = doc_id.wrapping_mul(7919).wrapping_add(seq as u64 + 1);
            let summary_id = split_id.wrapping_mul(1_000_003);

            self.put(
                CF_SUMMARIES,
                summary_id,
                &Summary::new(
                    summary_id,
                    doc_id,
                    split_id,
                    seq as i32,
                    &format!("summary of split {split_id}"),
                    7,
                    1.0,
                ),
            );
            self.put(
                CF_SPLITS,
                split_id,
                &Split::new(
                    split_id,
                    seq as i32,
                    doc_id,
                    &format!("split {seq} of document {doc_id}"),
                    42 + seq,
                    Some(vec![summary_id]),
                ),
            );

            if with_embeddings {
                self.put(
                    CF_EMBEDDINGS,
                    split_id,
                    &Embedding::new(
                        split_id,
                        EmbeddingDataType::Split,
                        vector(split_id),
                        model_id,
                    ),
                );
                self.put(
                    CF_EMBEDDINGS,
                    summary_id,
                    &Embedding::new(
                        summary_id,
                        EmbeddingDataType::Summary,
                        vector(summary_id),
                        model_id,
                    ),
                );
            }

            split_ids.push(split_id);
            summary_ids.push(summary_id);
        }

        // The old layer kept a document-level summary list as well as the per-split one. The two
        // name the same summaries here, which is what makes the union in decision D7 visible.
        let mut document = Document::new(&format!("test://doc/{doc_id}"), doc_id);
        document.split_ids = split_ids.clone();
        document.summary_ids = Some(summary_ids.clone());
        self.put(CF_DOCUMENTS, doc_id, &document);

        split_ids.into_iter().chain(summary_ids).collect()
    }

    /// One filter row per vector, which is the only record of what workspace a document is in.
    fn attach(&self, doc_id: u64, vector_ids: &[u64], workspace: Uuid) {
        for id in vector_ids {
            self.put(
                CF_FILTER_INFO,
                *id,
                &EmbeddingFilterInfo::new(*id, doc_id, workspace),
            );
        }
    }
}

fn options(source: &Path, destination: &Path) -> MigrationOptions {
    MigrationOptions {
        source: source.to_path_buf(),
        destination: destination.to_path_buf(),
        shard: ShardOptions::new(MODEL_ID, index_config()),
        memory_budget_mb: 64,
        dry_run: false,
        read_write: false,
    }
}

#[test]
fn each_workspace_becomes_its_own_shard() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace_a = Uuid::new_v4();
    let workspace_b = Uuid::new_v4();

    let source = Source::create(&src);
    source.attach(1, &source.write_document(1, 2, MODEL_ID, true), workspace_a);
    source.attach(2, &source.write_document(2, 1, MODEL_ID, true), workspace_a);
    source.attach(3, &source.write_document(3, 1, MODEL_ID, true), workspace_b);
    drop(source);

    let report = migrate(&options(&src, &dst)).unwrap();

    assert!(report.failures.is_empty(), "{:?}", report.failures);
    // Two vectors per split: the split's own and its summary's.
    assert_eq!(report.filter_rows, 8);
    assert_eq!(report.distinct_documents, 3);
    assert_eq!(report.shared_documents, 0);
    assert_eq!(report.orphan_documents, 0);
    assert_eq!(report.workspaces.len(), 2);
    assert_eq!(report.totals.docs, 3);
    assert_eq!(report.totals.splits, 4);
    assert_eq!(report.totals.summaries, 4);
    assert_eq!(report.shards_snapshotted, 2);

    for workspace in &report.workspaces {
        assert_eq!(
            workspace.written,
            Some(workspace.read),
            "shard for {} does not hold what was read",
            workspace.workspace
        );
    }
}

#[test]
fn a_document_in_two_workspaces_is_written_to_both() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace_a = Uuid::new_v4();
    let workspace_b = Uuid::new_v4();

    // A filter row is keyed by vector id, so a document cannot simply be attached twice: the
    // second attach would overwrite the first. Two workspaces claim one document only when its
    // own vectors' rows disagree, which is what a half-overwritten document in the old store
    // looks like. Here the splits say one workspace and the summaries say the other.
    let source = Source::create(&src);
    let vectors = source.write_document(1, 2, MODEL_ID, true);
    let (splits, summaries) = vectors.split_at(2);
    source.attach(1, splits, workspace_a);
    source.attach(1, summaries, workspace_b);
    drop(source);

    let report = migrate(&options(&src, &dst)).unwrap();

    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(report.distinct_documents, 1);
    assert_eq!(report.shared_documents, 1);
    // One document read twice, because each shard has to be able to answer for it on its own.
    assert_eq!(report.totals.docs, 2);
    assert_eq!(report.workspaces.len(), 2);
    for workspace in &report.workspaces {
        assert_eq!(workspace.written.unwrap().docs, 1);
        assert_eq!(workspace.written.unwrap().splits, 2);
    }
}

#[test]
fn a_document_no_filter_row_names_is_left_behind() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace = Uuid::new_v4();

    let source = Source::create(&src);
    source.attach(1, &source.write_document(1, 1, MODEL_ID, true), workspace);
    source.write_document(2, 1, MODEL_ID, true); // never attached to a workspace
    drop(source);

    let report = migrate(&options(&src, &dst)).unwrap();

    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(report.orphan_documents, 1);
    assert_eq!(report.totals.docs, 1);
}

#[test]
fn a_missing_vector_fails_one_document_and_no_others() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace = Uuid::new_v4();

    let source = Source::create(&src);
    source.attach(1, &source.write_document(1, 1, MODEL_ID, true), workspace);
    source.attach(2, &source.write_document(2, 1, MODEL_ID, false), workspace);
    drop(source);

    let report = migrate(&options(&src, &dst)).unwrap();

    assert_eq!(report.failures.len(), 1);
    assert!(
        report.failures[0].contains("has no embedding"),
        "{}",
        report.failures[0]
    );
    // The good document still went through.
    assert_eq!(report.totals.docs, 1);
    assert_eq!(report.workspaces[0].written.unwrap().docs, 1);
}

#[test]
fn vectors_from_another_model_are_refused() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace = Uuid::new_v4();

    let source = Source::create(&src);
    source.attach(
        1,
        &source.write_document(1, 1, OTHER_MODEL_ID, true),
        workspace,
    );
    drop(source);

    let report = migrate(&options(&src, &dst)).unwrap();

    assert_eq!(report.failures.len(), 1);
    assert!(
        report.failures[0].contains("model"),
        "{}",
        report.failures[0]
    );
    assert_eq!(report.totals.docs, 0);
}

#[test]
fn migrating_twice_leaves_the_same_shard() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace = Uuid::new_v4();

    let source = Source::create(&src);
    source.attach(1, &source.write_document(1, 3, MODEL_ID, true), workspace);
    drop(source);

    let first = migrate(&options(&src, &dst)).unwrap();
    let second = migrate(&options(&src, &dst)).unwrap();

    assert!(second.failures.is_empty(), "{:?}", second.failures);
    // The document ids are the same, so the second run replaces rather than duplicates.
    assert_eq!(second.workspaces[0].written, first.workspaces[0].written);
    assert_eq!(second.workspaces[0].written.unwrap().docs, 1);
    assert_eq!(second.workspaces[0].written.unwrap().splits, 3);
}

#[test]
fn the_destination_reopens_from_its_snapshot() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace = Uuid::new_v4();

    let source = Source::create(&src);
    source.attach(1, &source.write_document(1, 2, MODEL_ID, true), workspace);
    drop(source);

    let report = migrate(&options(&src, &dst)).unwrap();
    let written = report.workspaces[0].written.unwrap();

    let pool = ShardPool::new(&dst, ShardOptions::new(MODEL_ID, index_config())).unwrap();
    let shard = pool.get(workspace).unwrap();
    let stats = shard.stats();

    assert_eq!(stats.docs, written.docs);
    assert_eq!(stats.splits, written.splits);
    assert_eq!(stats.summaries, written.summaries);
    // Migration snapshots, so the reopen reads the snapshot instead of replaying the log.
    assert!(stats.snapshot_seq > 0);
    assert_eq!(stats.split_index_size, written.splits);
}

#[test]
fn a_dry_run_writes_nothing() {
    let dir = TempDir::new().unwrap();
    let (src, dst) = (dir.path().join("rocks"), dir.path().join("shards"));
    let workspace = Uuid::new_v4();

    let source = Source::create(&src);
    source.attach(1, &source.write_document(1, 2, MODEL_ID, true), workspace);
    drop(source);

    let mut opts = options(&src, &dst);
    opts.dry_run = true;
    let report = migrate(&opts).unwrap();

    assert_eq!(report.totals.docs, 1);
    assert_eq!(report.totals.splits, 2);
    assert!(report.workspaces[0].written.is_none());
    assert!(!dst.exists(), "the destination was created on a dry run");
}
