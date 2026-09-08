//! Step 4: the housekeeping pass the server's timer runs.
//!
//! The claim worth testing is not that the functions get called but what the pass buys: after one
//! snapshot pass a restart loads its indexes instead of rebuilding them, and eviction on the way
//! out keeps what it evicted.

use embedding_common::config::{IndexConfig, MetricKind, ScalarKind, StorageConfig};
use embedding_common::prelude::*;
use embedding_server::maintenance::{maintenance_pass, MaintenancePolicy};
use embedding_store::prelude::{ShardOptions, ShardPool};
use std::time::Duration;
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

fn policy() -> MaintenancePolicy {
    MaintenancePolicy::from_config(&StorageConfig::default())
}

// ---------------------------------------------------------------------------

#[test]
fn a_snapshot_pass_is_what_makes_the_next_start_fast() {
    let dir = TempDir::new().unwrap();
    let workspaces: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();

    {
        let pool = pool(&dir);
        for (i, ws) in workspaces.iter().enumerate() {
            let shard = pool.get(*ws).unwrap();
            for id in 1..=3 {
                shard.insert(&build_doc(id + (i as u64 * 100), 2)).unwrap();
            }
        }

        // An fsync-only pass makes the log durable but writes no derived files, so a restart would
        // still have the whole log to replay.
        maintenance_pass(&pool, &policy(), false);
        for ws in &workspaces {
            assert_eq!(pool.peek(*ws).unwrap().stats().snapshot_seq, 0);
        }

        maintenance_pass(&pool, &policy(), true);
        for ws in &workspaces {
            let stats = pool.peek(*ws).unwrap().stats();
            assert_eq!(stats.snapshot_seq, stats.next_seq - 1);
        }
    }

    let pool = pool(&dir);
    for ws in &workspaces {
        let shard = pool.get(*ws).unwrap();
        assert!(
            shard.loaded_from_snapshot(),
            "the pass should have left every shard with indexes to load"
        );
        assert_eq!(shard.stats().docs, 3);
    }
}

#[test]
fn the_pass_holds_the_loaded_set_to_the_budget() {
    let dir = TempDir::new().unwrap();
    let workspaces: Vec<Uuid> = (0..3).map(|_| Uuid::new_v4()).collect();
    let pool = ShardPool::new(
        dir.path().join("storage"),
        ShardOptions::new(MODEL_ID, index_config()),
    )
    .unwrap()
    .with_memory_budget_bytes(0);

    for ws in &workspaces {
        pool.get(*ws).unwrap().insert(&build_doc(1, 2)).unwrap();
    }
    let loaded_bytes = pool.stats().memory_bytes;

    maintenance_pass(&pool, &policy(), true);
    let stats = pool.stats();
    assert_eq!(stats.loaded, 3, "the shards stay open");
    assert_eq!(
        stats.viewed, 3,
        "but hand their graphs back to the page cache"
    );
    assert!(
        stats.memory_bytes < loaded_bytes,
        "demotion should cost less than loading: {} against {}",
        stats.memory_bytes,
        loaded_bytes
    );

    // A pass over shards that are already mapped has nothing left to give back.
    maintenance_pass(&pool, &policy(), true);
    assert_eq!(pool.stats().viewed, 3);

    // Each was snapshotted before its graphs went, so closing and reopening still loads.
    for ws in &workspaces {
        assert!(pool.close(*ws).unwrap());
        assert!(pool.get(*ws).unwrap().loaded_from_snapshot());
    }
}

#[test]
fn a_pass_over_an_empty_pool_does_nothing() {
    let dir = TempDir::new().unwrap();
    let pool = pool(&dir);
    maintenance_pass(&pool, &policy(), true);
    assert_eq!(pool.stats().loaded, 0);
    assert!(pool.list_workspaces().unwrap().is_empty());
}

#[test]
fn the_tick_never_outruns_the_shorter_interval() {
    let storage = StorageConfig {
        fsync_interval_ms: 1000,
        snapshot_interval_s: 300,
        ..StorageConfig::default()
    };
    assert_eq!(
        MaintenancePolicy::from_config(&storage).tick(),
        Duration::from_secs(1)
    );

    // Per-request fsync leaves the snapshot as the only reason to wake up.
    let storage = StorageConfig {
        fsync_interval_ms: 0,
        snapshot_interval_s: 60,
        ..StorageConfig::default()
    };
    let policy = MaintenancePolicy::from_config(&storage);
    assert_eq!(policy.fsync_interval, None);
    assert_eq!(policy.tick(), Duration::from_secs(60));

    // A snapshot interval shorter than the fsync interval pulls the tick down with it.
    let storage = StorageConfig {
        fsync_interval_ms: 5000,
        snapshot_interval_s: 1,
        ..StorageConfig::default()
    };
    assert_eq!(
        MaintenancePolicy::from_config(&storage).tick(),
        Duration::from_secs(1)
    );
}
