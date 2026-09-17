//! Shard tests: the shard, its two indexes, and the query read path.
//!
//! The store's own guarantees are covered by `store_tests.rs`; what matters here is that the
//! indexes stay in step with the log through inserts, replacements and deletes, that a snapshot
//! is read back instead of rebuilt, and that the read path shapes documents the way the query
//! handler expects.

use embedding_common::config::{IndexConfig, MetricKind, ScalarKind};
use embedding_common::prelude::*;
use embedding_store::prelude::*;
use indexmap::IndexMap;
use std::collections::HashSet;
use tempfile::TempDir;

const N_EMBD: usize = 16;
const MODEL_ID: u64 = 0xA11CE;

fn index_config() -> IndexConfig {
    IndexConfig {
        // A shard keeps its index files next to its log, so this is never read.
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

/// A deterministic unit vector. Different salts give well-separated directions, so the nearest
/// neighbour of a stored vector is that vector and nothing else.
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

/// Mix a document's salt with an entity id. Every entity has to get its own direction: two
/// entities sharing a vector would tie in the index, and a tie is not something a search is
/// entitled to break the same way twice.
fn seed(salt: u64, id: u64) -> u64 {
    salt.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ id.wrapping_mul(0xD1B5_4A32_D192_ED03)
}

fn summary(doc_id: u64, split_id: u64, sent_seq: i32, salt: u64) -> SummaryDto {
    let summary_id = split_id
        .wrapping_mul(1_000_003)
        .wrapping_add(sent_seq as u64 + 1);
    SummaryDto::new(
        summary_id,
        doc_id,
        split_id,
        sent_seq,
        &format!("summary {} of split {}", sent_seq, split_id),
        7 + sent_seq as usize,
        1.0 / (sent_seq as f32 + 1.0),
        Some(EmbeddingDto::new(
            summary_id,
            vector(seed(salt, summary_id)),
            MODEL_ID,
        )),
        None,
        None,
    )
}

/// A document whose document-level list is the union of its splits' lists.
fn build_doc(doc_id: u64, n_splits: usize, per_split: usize, salt: u64) -> DocumentDto {
    let mut splits = Vec::new();
    for seq in 0..n_splits {
        let split_id = doc_id.wrapping_mul(7919).wrapping_add(seq as u64 + 1);
        let summaries: Vec<SummaryDto> = (0..per_split)
            .map(|s| summary(doc_id, split_id, s as i32, salt))
            .collect();
        splits.push(SplitDto::new(
            split_id,
            seq as i32,
            doc_id,
            &format!("split {} of document {}", seq, doc_id),
            42 + seq,
            summaries,
            Some(EmbeddingDto::new(
                split_id,
                vector(seed(salt, split_id)),
                MODEL_ID,
            )),
            None,
            None,
        ));
    }
    let doc_summaries: Vec<SummaryDto> = splits.iter().flat_map(|s| s.summaries.clone()).collect();
    DocumentDto::new(
        doc_id,
        &format!("bench://doc/{}", doc_id),
        splits,
        Some(doc_summaries),
    )
}

fn embedding_of(split: &SplitDto) -> Vec<f32> {
    split.embedding.as_ref().unwrap().embedding.clone()
}

// ---------------------------------------------------------------------------

#[test]
fn search_finds_the_vector_that_was_inserted() {
    let dir = TempDir::new().unwrap();
    let shard = Shard::open(dir.path(), shard_options()).unwrap();

    let docs: Vec<DocumentDto> = (1..=5).map(|i| build_doc(i, 3, 2, i * 17)).collect();
    for doc in &docs {
        shard.insert(doc).unwrap();
    }

    let stats = shard.stats();
    assert_eq!(stats.docs, 5);
    assert_eq!(stats.splits, 15);
    assert_eq!(stats.summaries, 30);
    assert_eq!(stats.split_index_size, 15);
    assert_eq!(stats.summary_index_size, 30);

    let wanted = &docs[2].splits[1];
    let hits = shard.search_splits(&embedding_of(wanted), 3, None).unwrap();
    assert_eq!(
        *hits.keys().next().unwrap(),
        wanted.split_id,
        "the nearest split to a stored vector is that split"
    );
    assert!(hits[&wanted.split_id] < 1e-2, "distance to itself is ~0");
    let distances: Vec<f32> = hits.values().copied().collect();
    assert!(
        distances.windows(2).all(|w| w[0] <= w[1]),
        "hits come back ascending by distance: {:?}",
        distances
    );

    let wanted_summary = &docs[0].splits[0].summaries[1];
    let summary_hits = shard
        .search_summaries(
            &wanted_summary.embedding.as_ref().unwrap().embedding,
            3,
            None,
        )
        .unwrap();
    assert_eq!(
        *summary_hits.keys().next().unwrap(),
        wanted_summary.summary_id
    );
}

#[test]
fn reopening_without_a_snapshot_rebuilds_the_indexes() {
    let dir = TempDir::new().unwrap();
    let docs: Vec<DocumentDto> = (1..=4).map(|i| build_doc(i, 2, 2, i * 31)).collect();
    {
        let shard = Shard::open(dir.path(), shard_options()).unwrap();
        for doc in &docs {
            shard.insert(doc).unwrap();
        }
    }

    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    assert!(
        !shard.loaded_from_snapshot(),
        "nothing was saved, so the indexes come from the log"
    );
    assert_eq!(shard.stats().split_index_size, 8);

    let wanted = &docs[3].splits[0];
    let hits = shard.search_splits(&embedding_of(wanted), 1, None).unwrap();
    assert_eq!(*hits.keys().next().unwrap(), wanted.split_id);
}

#[test]
fn a_snapshot_is_read_back_instead_of_rebuilt() {
    let dir = TempDir::new().unwrap();
    let docs: Vec<DocumentDto> = (1..=4).map(|i| build_doc(i, 3, 2, i * 13)).collect();
    {
        let shard = Shard::open(dir.path(), shard_options()).unwrap();
        for doc in &docs {
            shard.insert(doc).unwrap();
        }
        let seq = shard.snapshot().unwrap();
        assert_eq!(seq, shard.stats().snapshot_seq);
    }

    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    assert!(
        shard.loaded_from_snapshot(),
        "the saved indexes describe exactly this log"
    );
    assert_eq!(shard.stats().splits, 12);
    assert_eq!(shard.stats().split_index_size, 12);

    let wanted = &docs[1].splits[2];
    let hits = shard.search_splits(&embedding_of(wanted), 1, None).unwrap();
    assert_eq!(*hits.keys().next().unwrap(), wanted.split_id);

    // A second snapshot with nothing written in between is a no-op that keeps the fast path.
    shard.snapshot().unwrap();
    drop(shard);
    assert!(Shard::open(dir.path(), shard_options())
        .unwrap()
        .loaded_from_snapshot());
}

#[test]
fn records_written_after_the_snapshot_force_a_rebuild() {
    let dir = TempDir::new().unwrap();
    let late = build_doc(99, 2, 1, 7);
    {
        let shard = Shard::open(dir.path(), shard_options()).unwrap();
        shard.insert(&build_doc(1, 2, 2, 3)).unwrap();
        shard.snapshot().unwrap();
        // The tail is in the log and in the maps, but not in any saved index, and nothing records
        // which entities it touched.
        shard.insert(&late).unwrap();
        shard.fsync().unwrap();
    }

    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    assert!(!shard.loaded_from_snapshot());
    assert_eq!(shard.stats().splits, 4);
    assert_eq!(shard.stats().split_index_size, 4);
    let hits = shard
        .search_splits(&embedding_of(&late.splits[1]), 1, None)
        .unwrap();
    assert_eq!(
        *hits.keys().next().unwrap(),
        late.splits[1].split_id,
        "the document written after the snapshot is searchable again"
    );
}

#[test]
fn deleting_a_document_takes_it_out_of_both_indexes() {
    let dir = TempDir::new().unwrap();
    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    let doomed = build_doc(1, 2, 2, 5);
    let kept = build_doc(2, 2, 2, 9);
    shard.insert(&doomed).unwrap();
    shard.insert(&kept).unwrap();

    assert!(shard.delete(doomed.document_id).unwrap());
    assert!(!shard.delete(doomed.document_id).unwrap(), "already gone");

    let stats = shard.stats();
    assert_eq!(stats.docs, 1);
    assert_eq!(stats.split_index_size, 2);
    assert_eq!(stats.summary_index_size, 4);

    let hits = shard
        .search_splits(&embedding_of(&doomed.splits[0]), 5, None)
        .unwrap();
    assert!(
        !hits.contains_key(&doomed.splits[0].split_id),
        "a deleted split is not a search result"
    );
    let summary_hits = shard
        .search_summaries(
            &doomed.splits[0].summaries[0]
                .embedding
                .as_ref()
                .unwrap()
                .embedding,
            5,
            None,
        )
        .unwrap();
    assert!(!summary_hits.contains_key(&doomed.splits[0].summaries[0].summary_id));
}

#[test]
fn reinserting_a_url_replaces_vectors_and_drops_the_splits_that_went_away() {
    let dir = TempDir::new().unwrap();
    let shard = Shard::open(dir.path(), shard_options()).unwrap();

    let first = build_doc(1, 3, 2, 21);
    shard.insert(&first).unwrap();
    let orphaned = first.splits[2].split_id;
    let reused = first.splits[0].split_id;

    // Same document, shorter and re-embedded: one split id disappears, the others are reused.
    let second = build_doc(1, 2, 2, 88);
    assert_eq!(second.splits[0].split_id, reused);
    let replaced = shard.insert(&second).unwrap();
    assert!(replaced.existed);
    assert!(
        replaced.split_ids.contains(&orphaned),
        "only the ids the new document does not reuse are reported"
    );
    assert!(!replaced.split_ids.contains(&reused));

    let stats = shard.stats();
    assert_eq!(stats.splits, 2);
    assert_eq!(stats.split_index_size, 2);

    let hits = shard
        .search_splits(&embedding_of(&first.splits[2]), 5, None)
        .unwrap();
    assert!(
        !hits.contains_key(&orphaned),
        "the split that went away is out of the index"
    );

    // The reused id now answers for the new vector, not the old one.
    let new_hits = shard
        .search_splits(&embedding_of(&second.splits[0]), 1, None)
        .unwrap();
    assert_eq!(*new_hits.keys().next().unwrap(), reused);
    assert!(new_hits[&reused] < 1e-2);
    let old_distance = shard
        .search_splits(&embedding_of(&first.splits[0]), 5, None)
        .unwrap()
        .get(&reused)
        .copied();
    assert!(
        old_distance.is_none_or(|d| d > 1e-2),
        "the old vector is no longer what that id holds"
    );
}

#[test]
fn a_document_filter_matches_only_those_documents() {
    let dir = TempDir::new().unwrap();
    let docs: Vec<DocumentDto> = (1..=6).map(|i| build_doc(i, 3, 2, i * 41)).collect();
    let query = vector(1234);
    let wanted: HashSet<u64> = [docs[1].document_id, docs[4].document_id]
        .into_iter()
        .collect();

    let scored_by_hand = |shard: &Shard| {
        let splits = shard.search_splits(&query, 4, Some(&wanted)).unwrap();
        let summaries = shard.search_summaries(&query, 4, Some(&wanted)).unwrap();
        (splits, summaries)
    };

    // Few enough candidates to score directly.
    let (brute_splits, brute_summaries) = {
        let shard = Shard::open(dir.path(), shard_options()).unwrap();
        for doc in &docs {
            shard.insert(doc).unwrap();
        }
        assert!(shard.options().brute_force_max >= 6);
        let out = scored_by_hand(&shard);
        shard.snapshot().unwrap();
        out
    };

    assert_eq!(brute_splits.len(), 4);
    for split_id in brute_splits.keys() {
        let doc_id = docs
            .iter()
            .find(|d| d.splits.iter().any(|s| s.split_id == *split_id))
            .unwrap()
            .document_id;
        assert!(
            wanted.contains(&doc_id),
            "a filtered search stays inside the filter"
        );
    }
    assert_eq!(brute_summaries.len(), 4);

    // The same filter over the graph instead: same shard, same vectors, threshold lowered so the
    // candidate count can never fall under it.
    let mut graph_options = shard_options();
    graph_options.brute_force_max = 0;
    let shard = Shard::open(dir.path(), graph_options).unwrap();
    assert!(shard.loaded_from_snapshot());
    let (graph_splits, graph_summaries) = scored_by_hand(&shard);

    assert_eq!(
        brute_splits.keys().collect::<Vec<_>>(),
        graph_splits.keys().collect::<Vec<_>>(),
        "the two filter paths agree on which splits are nearest"
    );
    assert_eq!(
        brute_summaries.keys().collect::<Vec<_>>(),
        graph_summaries.keys().collect::<Vec<_>>()
    );
    for (id, distance) in &brute_splits {
        assert!(
            (graph_splits[id] - distance).abs() < 1e-3,
            "and on how far away they are"
        );
    }

    // An empty filter is a filter, not the absence of one.
    assert!(shard
        .search_splits(&query, 4, Some(&HashSet::new()))
        .unwrap()
        .is_empty());
}

#[test]
fn the_query_read_path_carries_distances_and_only_the_summaries_that_were_hit() {
    let dir = TempDir::new().unwrap();
    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    let doc = build_doc(1, 3, 2, 77);
    shard.insert(&doc).unwrap();

    // A summary hit on split 0, a direct hit on split 1: the two ways into the answer.
    let hit_summary = &doc.splits[0].summaries[1];
    let summary_hits = shard
        .search_summaries(&hit_summary.embedding.as_ref().unwrap().embedding, 1, None)
        .unwrap();
    let summaries = shard
        .load_summaries(
            &summary_hits.keys().copied().collect::<Vec<_>>(),
            Some(&summary_hits),
            false,
        )
        .unwrap();
    assert_eq!(summaries.len(), 1);
    assert_eq!(summaries[0].summary_id, hit_summary.summary_id);
    assert_eq!(
        summaries[0].query_distance,
        Some(summary_hits[&hit_summary.summary_id])
    );
    assert!(
        summaries[0].embedding.is_none(),
        "embeddings were not asked for"
    );

    let mut hit_summaries: IndexMap<u64, Vec<SummaryDto>> = IndexMap::new();
    hit_summaries.insert(doc.splits[0].split_id, summaries);

    let direct = shard
        .search_splits(&embedding_of(&doc.splits[1]), 1, None)
        .unwrap();
    let order = vec![doc.splits[0].split_id, doc.splits[1].split_id];
    let splits = shard
        .load_splits(&order, Some(&direct), Some(&hit_summaries), false)
        .unwrap();

    assert_eq!(
        splits.iter().map(|s| s.split_id).collect::<Vec<_>>(),
        order,
        "splits come back in the order they were asked for"
    );
    assert_eq!(
        splits[0].query_distance, None,
        "a summary-derived split has no distance of its own"
    );
    assert_eq!(
        splits[0].summaries.len(),
        1,
        "only the summary that was hit"
    );
    assert_eq!(splits[0].summaries[0].summary_id, hit_summary.summary_id);
    assert_eq!(
        splits[1].query_distance,
        Some(direct[&doc.splits[1].split_id])
    );
    assert!(
        splits[1].summaries.is_empty(),
        "a direct hit carries no summaries"
    );

    // The document keeps its own complete list, in recorded order and without distances.
    let loaded = shard
        .load_doc(doc.document_id, splits, false)
        .unwrap()
        .unwrap();
    assert_eq!(loaded.document_url, doc.document_url);
    assert_eq!(loaded.splits.len(), 2);
    let doc_summaries = loaded.summaries.unwrap();
    assert_eq!(
        doc_summaries
            .iter()
            .map(|s| s.summary_id)
            .collect::<Vec<_>>(),
        doc.summaries
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| s.summary_id)
            .collect::<Vec<_>>()
    );
    assert!(doc_summaries.iter().all(|s| s.query_distance.is_none()));

    assert!(shard.load_doc(404, Vec::new(), false).unwrap().is_none());
}

#[test]
fn retrieval_returns_the_document_fully_nested() {
    let dir = TempDir::new().unwrap();
    let shard = Shard::open(dir.path(), shard_options()).unwrap();
    let doc = build_doc(1, 2, 2, 55);
    shard.insert(&doc).unwrap();

    assert_eq!(
        shard.doc_id_for_url(&doc.document_url),
        Some(doc.document_id)
    );
    assert!(shard.contains_doc(doc.document_id));

    let loaded = shard.get_doc(doc.document_id, true).unwrap().unwrap();
    assert_eq!(loaded.splits.len(), 2);
    for (got, want) in loaded.splits.iter().zip(doc.splits.iter()) {
        assert_eq!(got.split_id, want.split_id);
        assert_eq!(got.text_content, want.text_content);
        assert_eq!(got.summaries.len(), want.summaries.len());
        assert!(got.query_distance.is_none());
        let stored = &got.embedding.as_ref().unwrap().embedding;
        let original = want.embedding.as_ref().unwrap();
        for (a, b) in stored.iter().zip(original.embedding.iter()) {
            assert!((a - b).abs() < 1e-2, "f16 round trip: {} vs {}", a, b);
        }
    }
    assert_eq!(loaded.summaries.unwrap().len(), 4);

    let entry = shard.doc_entry(doc.document_id).unwrap();
    assert_eq!(entry.split_ids.len(), 2);
    assert_eq!(entry.summary_ids.as_ref().unwrap().len(), 4);
    assert!(
        entry.extra_summary_ids.is_empty(),
        "the document list is the union"
    );
}

#[test]
fn compaction_leaves_the_indexes_and_the_snapshot_path_intact() {
    let dir = TempDir::new().unwrap();
    let mut options = shard_options();
    // Small enough that the log runs to several segments, so compaction really moves records and
    // rewrites the locations the brute-force path reads vectors through.
    options.segment_max_bytes = 4 * 1024;

    let docs: Vec<DocumentDto> = (1..=40u64).map(|i| build_doc(i, 3, 2, i)).collect();
    let shard = Shard::open(dir.path(), options.clone()).unwrap();
    for doc in &docs {
        shard.insert(doc).unwrap();
    }
    for doc in docs.iter().filter(|d| d.document_id % 2 == 0) {
        assert!(shard.delete(doc.document_id).unwrap());
    }
    shard.seal_active().unwrap();
    assert!(shard.stats().tombstone_bytes > 0);

    assert!(
        shard.compact_if_needed(0.1).unwrap(),
        "half the log is dead"
    );
    let stats = shard.stats();
    assert_eq!(stats.docs, 20);
    assert_eq!(stats.tombstone_bytes, 0);
    assert_eq!(stats.split_index_size, 60);

    // Vectors are read through locations compaction rewrote: an unfiltered search walks the graph,
    // a filtered one scores the records directly, and both have to still find the same split.
    let survivor = &docs[10];
    assert_eq!(survivor.document_id % 2, 1);
    let query = embedding_of(&survivor.splits[1]);
    let hits = shard.search_splits(&query, 1, None).unwrap();
    assert_eq!(*hits.keys().next().unwrap(), survivor.splits[1].split_id);

    let filter: HashSet<u64> = [survivor.document_id].into_iter().collect();
    let filtered = shard.search_splits(&query, 3, Some(&filter)).unwrap();
    assert_eq!(
        *filtered.keys().next().unwrap(),
        survivor.splits[1].split_id
    );
    assert!(filtered[&survivor.splits[1].split_id] < 1e-2);

    // The snapshot compaction takes on the way out is the one the next open reads back.
    drop(shard);
    let shard = Shard::open(dir.path(), options).unwrap();
    assert!(shard.loaded_from_snapshot());
    assert_eq!(shard.stats().docs, 20);
    assert_eq!(
        *shard
            .search_splits(&query, 1, None)
            .unwrap()
            .keys()
            .next()
            .unwrap(),
        survivor.splits[1].split_id
    );
}
