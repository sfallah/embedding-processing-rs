//! The `embedding-index` tests, ported to the shard.
//!
//! What survives the port is what was actually being claimed there and is not claimed elsewhere in
//! this crate: that the index is built with the parameters the config gives and keeps them when it
//! is read back, that a vector goes into the log and into the graph as the same f16 and comes back
//! that way, and that a saved index can be reopened, written to, and saved again. Workspace
//! filtering moved to `pool_tests.rs`, because a workspace is now a shard rather than a predicate,
//! and the document filter is covered in `shard_tests.rs`.

use embedding_common::config::{IndexConfig, MetricKind, ScalarKind};
use embedding_common::prelude::*;
use embedding_store::prelude::*;
use embedding_store::vector_index::VectorIndex;
use half::f16;
use tempfile::TempDir;

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

fn build_doc(doc_id: u64, n_splits: usize, salt: u64) -> DocumentDto {
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
            Some(EmbeddingDto::new(
                summary_id,
                vector(summary_id ^ salt),
                MODEL_ID,
            )),
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
            Some(EmbeddingDto::new(
                split_id,
                vector(split_id ^ salt),
                MODEL_ID,
            )),
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
fn the_index_keeps_the_configured_parameters_across_a_save_and_load() {
    let dir = TempDir::new().unwrap();
    let config = index_config();

    let index = VectorIndex::create(dir.path(), "splits", &config).unwrap();
    assert_eq!(index.dimensions(), config.dimensions);
    assert_eq!(index.connectivity(), config.connectivity);
    assert_eq!(index.expansion_add(), config.expansion_add);
    assert_eq!(index.expansion_search(), config.expansion_search);

    for id in 1..=32u64 {
        index.upsert(id, &vector(id)).unwrap();
    }
    index.save_atomically().unwrap();

    // Creating from the config and then loading is what `Shard::open` does. `Index::restore` would
    // read the same file but come back with the library's defaults (128 and 64 on 2.26.2) instead
    // of these, which is why nothing here uses it.
    let reopened = VectorIndex::create(dir.path(), "splits", &config).unwrap();
    reopened.load().unwrap();
    assert_eq!(reopened.size(), 32);
    assert_eq!(reopened.connectivity(), config.connectivity);
    assert_eq!(reopened.expansion_add(), config.expansion_add);
    assert_eq!(reopened.expansion_search(), config.expansion_search);
}

#[test]
fn the_log_and_the_graph_hold_the_same_f16_vector() {
    let dir = TempDir::new().unwrap();
    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    let docs: Vec<DocumentDto> = (1..=3).map(|i| build_doc(i, 3, i * 29)).collect();
    for doc in &docs {
        shard.insert(doc).unwrap();
    }

    for doc in &docs {
        for split in &doc.splits {
            let original = &split.embedding.as_ref().unwrap().embedding;

            // The record gives back exactly what f16 can represent, not merely something near it.
            let loaded = shard
                .load_splits(&[split.split_id], None, None, true)
                .unwrap();
            let stored = &loaded[0].embedding.as_ref().unwrap().embedding;
            let expected: Vec<f32> = original
                .iter()
                .map(|x| f16::from_f32(*x).to_f32())
                .collect();
            assert_eq!(stored, &expected, "split {}", split.split_id);

            // And the graph holds that same vector, so a split finds itself first.
            let hits = shard.search_splits(original, 3, None).unwrap();
            assert_eq!(*hits.keys().next().unwrap(), split.split_id);
            assert!(
                hits[&split.split_id] < 1e-6,
                "a split should be its own nearest neighbour, distance was {}",
                hits[&split.split_id]
            );
        }
    }
}

#[test]
fn a_reloaded_shard_takes_more_writes_and_reloads_again() {
    let dir = TempDir::new().unwrap();
    let first: Vec<DocumentDto> = (1..=4).map(|i| build_doc(i, 3, i * 13)).collect();
    let second: Vec<DocumentDto> = (5..=7).map(|i| build_doc(i, 3, i * 17)).collect();

    {
        let shard = Shard::open(dir.path(), shard_options()).unwrap();
        for doc in &first {
            shard.insert(doc).unwrap();
        }
        shard.snapshot().unwrap();
    }

    {
        let shard = Shard::open(dir.path(), shard_options()).unwrap();
        assert!(shard.loaded_from_snapshot());
        assert_eq!(shard.stats().split_index_size, 12);
        // Writing to an index that came off disk, then saving it again, is the path a long-lived
        // server spends all its time on.
        for doc in &second {
            shard.insert(doc).unwrap();
        }
        assert_eq!(shard.stats().split_index_size, 21);
        shard.snapshot().unwrap();
    }

    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    assert!(shard.loaded_from_snapshot());
    let stats = shard.stats();
    assert_eq!(stats.docs, 7);
    assert_eq!(stats.split_index_size, 21);
    assert_eq!(stats.summary_index_size, 21);

    // Both generations are searchable, the one that was saved twice included.
    for doc in first.iter().chain(second.iter()) {
        let split = &doc.splits[1];
        let query = &split.embedding.as_ref().unwrap().embedding;
        let hits = shard.search_splits(query, 1, None).unwrap();
        assert_eq!(*hits.keys().next().unwrap(), split.split_id);
    }
}

#[test]
fn a_mapped_index_costs_a_fraction_of_a_resident_one() {
    // The reason a cold shard is demoted rather than dropped: usearch answers searches from the
    // mapped file, and what it holds resident to do so is a rounding error against the graph.
    let dir = TempDir::new().unwrap();
    let config = index_config();
    let index = VectorIndex::create(dir.path(), "splits", &config).unwrap();
    for id in 1..=4000u64 {
        index.upsert(id, &vector(id)).unwrap();
    }
    index.save_atomically().unwrap();

    let loaded = VectorIndex::create(dir.path(), "splits", &config).unwrap();
    loaded.load().unwrap();
    assert!(!loaded.is_viewed());
    let resident = loaded.memory_bytes();

    let viewed = VectorIndex::create(dir.path(), "splits", &config).unwrap();
    viewed.view().unwrap();
    assert!(viewed.is_viewed());
    let mapped = viewed.memory_bytes();

    assert_eq!(viewed.size(), 4000, "a mapped index knows what it holds");
    let hits = viewed.search(&vector(7), 1).unwrap();
    assert_eq!(hits[0].0, 7, "and answers searches from the mapping");

    assert!(
        mapped * 4 < resident,
        "mapping should cost far less than loading: {} against {}",
        mapped,
        resident
    );
}
