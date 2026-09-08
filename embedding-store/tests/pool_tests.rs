//! Step 3 tests: the pool that keeps one shard per workspace under a memory budget.
//!
//! What matters here is what the pool adds on top of a shard: that a workspace maps to exactly
//! one live `Shard` however many callers ask for it at once, that eviction goes through a
//! snapshot so a reload is not a rebuild, and that a shard someone is holding stays put.

use embedding_common::config::{IndexConfig, MetricKind, ScalarKind};
use embedding_common::prelude::*;
use embedding_store::prelude::*;
use std::sync::Arc;
use tempfile::TempDir;
use uuid::Uuid;

const N_EMBD: usize = 16;
const MODEL_ID: u64 = 0xA11CE;

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

fn pool(dir: &TempDir) -> ShardPool {
    ShardPool::new(
        dir.path().join("storage"),
        ShardOptions::new(MODEL_ID, index_config()),
    )
    .unwrap()
}

/// A deterministic unit vector, one direction per salt.
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

/// A document with `n_splits` splits, one summary each, as decision D7 shapes it.
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

// ---------------------------------------------------------------------------

#[test]
fn a_workspace_gets_its_own_directory_and_one_shard() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    let (a, b) = (Uuid::new_v4(), Uuid::new_v4());

    let first = pool.get(a).unwrap();
    let again = pool.get(a).unwrap();
    let other = pool.get(b).unwrap();

    assert!(Arc::ptr_eq(&first, &again), "one shard per workspace");
    assert!(!Arc::ptr_eq(&first, &other));
    assert_eq!(first.dir(), pool.shard_dir(a));
    assert!(pool.shard_dir(a).join("manifest").is_file());
    assert_eq!(pool.stats().loaded, 2);
}

#[test]
fn concurrent_gets_of_one_workspace_open_it_once() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    let ws = Uuid::new_v4();

    // The gate has to hold whichever threads lose the race, or they would each open a second
    // `Shard` on the same log and both would own its active segment.
    let shards: Vec<Arc<Shard>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| pool.get(ws).unwrap()))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    assert_eq!(pool.loaded().len(), 1);
    for shard in &shards {
        assert!(Arc::ptr_eq(shard, &shards[0]), "every caller got one shard");
    }
}

#[test]
fn eviction_snapshots_so_the_reload_is_not_a_rebuild() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    let ws = Uuid::new_v4();

    let shard = pool.get(ws).unwrap();
    for id in 1..=5 {
        shard.insert(&build_doc(id, 3)).unwrap();
    }
    assert!(
        !shard.loaded_from_snapshot(),
        "a fresh shard has nothing to load"
    );
    drop(shard);

    assert!(pool.evict(ws).unwrap());
    assert!(!pool.is_loaded(ws));
    assert_eq!(pool.stats().loaded, 0);

    let reloaded = pool.get(ws).unwrap();
    assert!(
        reloaded.loaded_from_snapshot(),
        "eviction has to snapshot, or every reload pays for a rebuild"
    );
    assert_eq!(reloaded.stats().docs, 5);
    assert_eq!(reloaded.stats().split_index_size, 15);
    let doc = reloaded
        .get_doc(3, false)
        .unwrap()
        .expect("document survived");
    assert_eq!(doc.splits.len(), 3);
}

#[test]
fn a_held_shard_is_never_evicted() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir).with_memory_budget_bytes(0);
    let ws = Uuid::new_v4();

    let held = pool.get(ws).unwrap();
    held.insert(&build_doc(1, 3)).unwrap();

    assert!(!pool.evict(ws).unwrap(), "a caller still holds it");
    assert_eq!(pool.enforce_budget().unwrap(), 0);
    assert!(pool.is_loaded(ws));

    // The budget only takes effect once the last holder is done with it.
    drop(held);
    assert_eq!(pool.enforce_budget().unwrap(), 1);
    assert!(!pool.is_loaded(ws));
}

#[test]
fn the_budget_evicts_the_least_recently_used_shard() {
    let dir = TempDir::new().unwrap();
    let (a, b) = (Uuid::new_v4(), Uuid::new_v4());
    let storage = dir.path().join("storage");

    // Two shards with identical content, so the budget can be set to fit exactly one of them.
    let one_shard_bytes = {
        let pool = ShardPool::new(&storage, ShardOptions::new(MODEL_ID, index_config())).unwrap();
        for ws in [a, b] {
            let shard = pool.get(ws).unwrap();
            for id in 1..=5 {
                shard.insert(&build_doc(id, 3)).unwrap();
            }
        }
        pool.snapshot_all().unwrap();
        let total = pool.stats().memory_bytes;
        assert!(total > 0, "an index with vectors in it reports residency");
        total / 2
    };

    let pool = ShardPool::new(&storage, ShardOptions::new(MODEL_ID, index_config()))
        .unwrap()
        .with_memory_budget_bytes(one_shard_bytes);
    let held_a = pool.get(a).unwrap();
    let held_b = pool.get(b).unwrap();
    // Nothing can be evicted while both are held, so both are loaded and over budget here.
    assert_eq!(pool.stats().loaded, 2);
    assert!(pool.stats().memory_bytes > pool.budget_bytes());

    pool.peek(a).expect("a is loaded"); // a becomes the most recently used
    drop(held_a);
    drop(held_b);

    assert_eq!(pool.enforce_budget().unwrap(), 1);
    assert!(pool.is_loaded(a), "the most recently used shard stays");
    assert!(!pool.is_loaded(b), "the least recently used shard goes");
}

#[test]
fn snapshot_and_fsync_visit_every_loaded_shard() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    let workspaces: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
    for ws in &workspaces {
        let shard = pool.get(*ws).unwrap();
        shard.insert(&build_doc(1, 2)).unwrap();
    }

    assert_eq!(pool.fsync_all().unwrap(), 3);
    assert_eq!(pool.snapshot_all().unwrap(), 3);
    for ws in &workspaces {
        let shard = pool.peek(*ws).unwrap();
        assert_eq!(shard.stats().snapshot_seq, shard.stats().next_seq - 1);
    }

    // Nothing has been written since, so a second pass is the cheap path rather than a rewrite.
    assert_eq!(pool.snapshot_all().unwrap(), 3);
}

#[test]
fn list_workspaces_reads_the_directory_not_the_loaded_set() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    let mut expected: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
    for ws in &expected {
        pool.get(*ws).unwrap();
    }
    expected.sort();

    // A shard that is on disk but not in memory still belongs to the workspace list.
    assert!(pool.evict(expected[0]).unwrap());
    std::fs::create_dir_all(pool.dir().join("not-a-workspace")).unwrap();
    std::fs::write(pool.dir().join("stray.txt"), b"ignored").unwrap();

    assert_eq!(pool.list_workspaces().unwrap(), expected);
    assert_eq!(pool.stats().loaded, 2);
}

#[test]
fn get_existing_does_not_bring_a_workspace_into_being() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    let ws = Uuid::new_v4();

    assert!(pool.get_existing(ws).unwrap().is_none());
    assert!(
        !pool.shard_dir(ws).exists(),
        "a read must not create a shard"
    );
    assert!(pool.list_workspaces().unwrap().is_empty());

    // Once an insert has created it, a read finds it, loaded or not.
    pool.get(ws).unwrap().insert(&build_doc(1, 2)).unwrap();
    assert!(pool.evict(ws).unwrap());
    let found = pool
        .get_existing(ws)
        .unwrap()
        .expect("the shard is on disk");
    assert_eq!(found.stats().docs, 1);
}

#[test]
fn evicting_an_unknown_workspace_is_not_an_error() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    assert!(!pool.evict(Uuid::new_v4()).unwrap());
    assert!(pool.list_workspaces().unwrap().is_empty());
}
