//! Step 4: what survives a process that dies without a chance to tidy up.
//!
//! The child half opens a shard, inserts, fsyncs, inserts more, and aborts — no snapshot, no
//! shutdown handler, no destructors. The parent then reopens the same directory and checks that
//! the shard puts itself back together from the log alone: every acknowledged document present and
//! searchable, and maps identical to a second, independent replay.
//!
//! The child is a `#[ignore]`d test in this same binary, re-executed by the parent. That keeps the
//! fixture in one file and needs no extra binary in the crate.

use embedding_common::config::{IndexConfig, MetricKind, ScalarKind};
use embedding_common::prelude::*;
use embedding_store::prelude::*;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const N_EMBD: usize = 16;
const MODEL_ID: u64 = 0xA11CE;
const CRASH_DIR: &str = "EMBEDDING_STORE_CRASH_DIR";
/// Documents written and fsynced before the child dies.
const ACKNOWLEDGED: u64 = 12;
/// Documents written after that fsync, still with no snapshot anywhere.
const AFTERWARDS: u64 = 5;

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

fn shard_options() -> ShardOptions {
    ShardOptions::new(MODEL_ID, index_config())
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

fn build_doc(doc_id: u64, n_splits: usize) -> DocumentDto {
    let mut splits = Vec::new();
    for seq in 0..n_splits {
        let split_id = doc_id.wrapping_mul(7919).wrapping_add(seq as u64 + 1);
        let summary_id = split_id.wrapping_mul(1_000_003);
        let summary = SummaryDto::new(
            summary_id,
            doc_id,
            split_id,
            0,
            &format!("summary of split {}", split_id),
            7,
            1.0,
            Some(EmbeddingDto::new(summary_id, vector(summary_id), MODEL_ID)),
            None,
            None,
        );
        splits.push(SplitDto::new(
            split_id,
            seq as i32,
            doc_id,
            &format!("split {} of document {}", seq, doc_id),
            42 + seq,
            vec![summary],
            Some(EmbeddingDto::new(split_id, vector(split_id), MODEL_ID)),
            None,
            None,
        ));
    }
    let doc_summaries: Vec<SummaryDto> = splits.iter().flat_map(|s| s.summaries.clone()).collect();
    DocumentDto::new(
        doc_id,
        &format!("test://doc/{}", doc_id),
        splits,
        Some(doc_summaries),
    )
}

/// Every document id the shard should hold, and the split ids under them.
fn expected_ids() -> (Vec<u64>, Vec<u64>) {
    let docs: Vec<u64> = (1..=ACKNOWLEDGED + AFTERWARDS).collect();
    let splits = docs
        .iter()
        .flat_map(|id| build_doc(*id, 3).splits.into_iter().map(|s| s.split_id))
        .collect();
    (docs, splits)
}

// ---------------------------------------------------------------------------

/// The child. Runs only when the parent puts a directory in the environment.
#[test]
#[ignore = "re-executed as a child by the crash test"]
fn crash_child() {
    let dir = std::env::var(CRASH_DIR).expect("the parent sets the shard directory");
    let shard = Shard::open(&dir, shard_options()).expect("child could not open the shard");

    for id in 1..=ACKNOWLEDGED {
        shard
            .insert(&build_doc(id, 3))
            .expect("child insert failed");
    }
    // The acknowledgement a client would have had.
    shard.fsync().expect("child fsync failed");
    println!("acknowledged {}", ACKNOWLEDGED);
    let _ = std::io::stdout().flush();

    // Written but never made durable, and no snapshot is taken at any point.
    for id in ACKNOWLEDGED + 1..=ACKNOWLEDGED + AFTERWARDS {
        shard
            .insert(&build_doc(id, 3))
            .expect("child insert failed");
    }

    // Die the way a killed process dies: no unwinding, no destructors, nothing flushed.
    std::process::abort();
}

#[test]
fn a_shard_rebuilds_itself_after_a_process_dies_without_a_snapshot() {
    let dir = TempDir::new().unwrap();
    let shard_dir = dir.path().join("shard");

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child", "--ignored", "--nocapture"])
        .env(CRASH_DIR, &shard_dir)
        .status()
        .expect("could not run the child");
    assert!(
        !status.success(),
        "the child was supposed to abort, not finish: {:?}",
        status
    );

    // Nothing tidied up on the way out.
    assert!(
        !shard_dir.join("meta.snap").exists(),
        "the child took no meta snapshot"
    );
    assert!(
        !shard_dir.join("indexes.state").exists(),
        "the child saved no indexes"
    );

    let shard = Shard::open(&shard_dir, shard_options()).expect("reopen after the crash");
    assert!(
        !shard.loaded_from_snapshot(),
        "there was no snapshot to load; the log is all there is"
    );

    let (doc_ids, split_ids) = expected_ids();
    let stats = shard.stats();
    assert_eq!(stats.docs, doc_ids.len());
    assert_eq!(stats.splits, split_ids.len());
    // The indexes were rebuilt from the log, so they hold exactly what the log does.
    assert_eq!(stats.split_index_size, split_ids.len());
    assert_eq!(stats.summary_index_size, split_ids.len());

    // Every document is readable and every split is searchable, acknowledged or not: a process
    // that dies does not take the page cache with it, so the whole log survived.
    for doc_id in &doc_ids {
        let doc = shard
            .get_doc(*doc_id, false)
            .unwrap()
            .unwrap_or_else(|| panic!("document {} is missing after the crash", doc_id));
        assert_eq!(doc.splits.len(), 3);
    }
    for doc_id in &doc_ids {
        let doc = build_doc(*doc_id, 3);
        let split = &doc.splits[1];
        let query = &split.embedding.as_ref().unwrap().embedding;
        let hits = shard.search_splits(query, 1, None).unwrap();
        assert_eq!(
            *hits.keys().next().unwrap(),
            split.split_id,
            "document {} is not searchable after the crash",
            doc_id
        );
    }

    // A second, independent replay of the same log has to agree with the first one.
    let again = Shard::open(&shard_dir, shard_options()).unwrap();
    assert_eq!(again.stats(), stats);
    for doc_id in &doc_ids {
        assert_eq!(shard.doc_entry(*doc_id), again.doc_entry(*doc_id));
    }
}

#[test]
fn the_shard_left_behind_is_usable_rather_than_merely_readable() {
    // Recovery is not finished if the shard cannot take writes again afterwards.
    let dir = TempDir::new().unwrap();
    let shard_dir = dir.path().join("shard");

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child", "--ignored", "--nocapture"])
        .env(CRASH_DIR, &shard_dir)
        .status()
        .expect("could not run the child");
    assert!(!status.success());

    let shard = Shard::open(&shard_dir, shard_options()).unwrap();
    let before = shard.stats().docs;

    let fresh = build_doc(9_000, 3);
    shard.insert(&fresh).unwrap();
    shard.delete(1).unwrap();
    assert_eq!(shard.stats().docs, before);

    let seq = shard.snapshot().unwrap();
    assert!(seq > 0);
    drop(shard);

    // And the snapshot it takes is one the next start can use.
    let reopened = Shard::open(&shard_dir, shard_options()).unwrap();
    assert!(reopened.loaded_from_snapshot());
    assert!(reopened.get_doc(1, false).unwrap().is_none());
    assert!(reopened.get_doc(9_000, false).unwrap().is_some());
}

/// A sanity check on the fixture: the child really does write into the directory it is given.
#[test]
fn the_child_writes_where_it_is_told() {
    let dir = TempDir::new().unwrap();
    let shard_dir: &Path = &dir.path().join("shard");
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "crash_child", "--ignored", "--nocapture"])
        .env(CRASH_DIR, shard_dir)
        .status()
        .unwrap();
    assert!(shard_dir.join("manifest").is_file());
}
