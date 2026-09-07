//! Step 2 tests: records, replay, sealing, compaction and the meta snapshot.
//!
//! Randomised sequences use a hand-rolled generator rather than a proptest dependency, so a
//! failure prints the seed and is replayed by hand.

use embedding_common::prelude::*;
use embedding_store::prelude::*;
use embedding_store::maps::Maps;
use embedding_store::meta_snapshot;
use tempfile::TempDir;

const N_EMBD: usize = 16;
const MODEL_ID: u64 = 0xA11CE;

fn opts() -> StoreOptions {
    StoreOptions::new(MODEL_ID, N_EMBD)
}

/// Deterministic, reproducible pseudo-randomness.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn vector(salt: u64) -> Vec<f32> {
    (0..N_EMBD)
        .map(|i| ((salt.wrapping_mul(31).wrapping_add(i as u64) % 1000) as f32) / 1000.0)
        .collect()
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
        Some(EmbeddingDto::new(summary_id, vector(salt), MODEL_ID)),
        None,
        None,
    )
}

/// How the document-level summary list is built.
enum DocList {
    /// Every split summary, in split order — decision D7, the production shape today.
    Union,
    /// Only the first summary of each split, so the list is not the union.
    FirstOfEachSplit,
    /// No document-level list at all.
    None,
}

fn build_doc(doc_id: u64, n_splits: usize, per_split: usize, list: DocList, salt: u64) -> DocumentDto {
    let mut splits = Vec::new();
    for seq in 0..n_splits {
        let split_id = doc_id.wrapping_mul(7919).wrapping_add(seq as u64 + 1);
        let summaries: Vec<SummaryDto> = (0..per_split)
            .map(|s| summary(doc_id, split_id, s as i32, salt + s as u64))
            .collect();
        splits.push(SplitDto::new(
            split_id,
            seq as i32,
            doc_id,
            &format!("split {} of document {}", seq, doc_id),
            42 + seq,
            summaries,
            Some(EmbeddingDto::new(split_id, vector(salt + seq as u64), MODEL_ID)),
            None,
            None,
        ));
    }

    let doc_summaries = match list {
        DocList::Union => Some(splits.iter().flat_map(|s| s.summaries.clone()).collect()),
        DocList::FirstOfEachSplit => Some(
            splits
                .iter()
                .filter_map(|s| s.summaries.first().cloned())
                .collect(),
        ),
        DocList::None => None,
    };

    DocumentDto::new(
        doc_id,
        &format!("bench://doc/{}", doc_id),
        splits,
        doc_summaries,
    )
}

fn assert_vectors_close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-2, "f16 round trip lost too much: {} vs {}", x, y);
    }
}

// ---------------------------------------------------------------------------

#[test]
fn every_record_kind_round_trips() {
    let dir = TempDir::new().unwrap();
    let mut store = Store::open(dir.path(), opts()).unwrap();

    let doc = build_doc(1, 3, 2, DocList::Union, 11);
    store.insert(&doc).unwrap();

    // Document, fully nested and with embeddings.
    let loaded = store.get_doc(1, true).unwrap().unwrap();
    assert_eq!(loaded.document_id, doc.document_id);
    assert_eq!(loaded.document_url, doc.document_url);
    assert_eq!(loaded.splits.len(), 3);
    for (got, want) in loaded.splits.iter().zip(doc.splits.iter()) {
        assert_eq!(got.split_id, want.split_id);
        assert_eq!(got.sequence_id, want.sequence_id);
        assert_eq!(got.doc_id, want.doc_id);
        assert_eq!(got.text_content, want.text_content);
        assert_eq!(got.token_len, want.token_len);
        assert_eq!(got.summaries.len(), want.summaries.len());
        assert_vectors_close(
            &got.embedding.as_ref().unwrap().embedding,
            &want.embedding.as_ref().unwrap().embedding,
        );
        for (gs, ws) in got.summaries.iter().zip(want.summaries.iter()) {
            assert_eq!(gs.summary_id, ws.summary_id);
            assert_eq!(gs.text_content, ws.text_content);
            assert_eq!(gs.split_sequence_id, ws.split_sequence_id);
            assert_eq!(gs.token_len, ws.token_len);
            assert!((gs.centrality - ws.centrality).abs() < 1e-6);
            assert_vectors_close(
                &gs.embedding.as_ref().unwrap().embedding,
                &ws.embedding.as_ref().unwrap().embedding,
            );
        }
    }

    // Single split and single summary reads.
    let split_id = doc.splits[0].split_id;
    let split = store.get_split(split_id, false).unwrap().unwrap();
    assert_eq!(split.text_content, doc.splits[0].text_content);
    assert!(split.embedding.is_none(), "embeddings were not asked for");
    let summary_id = doc.splits[0].summaries[0].summary_id;
    let summary = store.get_summary(summary_id, false).unwrap().unwrap();
    assert_eq!(summary.split_id, split_id);

    // DeleteDoc.
    let removed = store.delete(1).unwrap().unwrap();
    assert_eq!(removed.split_ids.len(), 3);
    assert_eq!(removed.summary_ids.len(), 6);
    assert!(store.get_doc(1, false).unwrap().is_none());
    assert!(store.get_split(split_id, false).unwrap().is_none());
    assert!(store.get_summary(summary_id, false).unwrap().is_none());
}

#[test]
fn document_list_that_is_the_union_stores_each_summary_once() {
    let dir = TempDir::new().unwrap();
    let mut store = Store::open(dir.path(), opts()).unwrap();

    let doc = build_doc(2, 4, 2, DocList::Union, 5);
    store.insert(&doc).unwrap();

    // Eight distinct summaries, each stored once even though every one is in two lists.
    assert_eq!(store.summary_count(), 8);
    assert_eq!(store.split_count(), 4);

    let loaded = store.get_doc(2, false).unwrap().unwrap();
    let doc_list = loaded.summaries.as_ref().expect("document list");
    assert_eq!(doc_list.len(), 8);

    // The document list comes back in the order it was recorded, split by split.
    let expected: Vec<u64> = doc
        .summaries
        .as_ref()
        .unwrap()
        .iter()
        .map(|s| s.summary_id)
        .collect();
    let got: Vec<u64> = doc_list.iter().map(|s| s.summary_id).collect();
    assert_eq!(got, expected);

    // And each split still carries its own list.
    for split in &loaded.splits {
        assert_eq!(split.summaries.len(), 2);
    }
}

#[test]
fn document_list_that_differs_from_the_union_round_trips() {
    let dir = TempDir::new().unwrap();
    let mut store = Store::open(dir.path(), opts()).unwrap();

    let doc = build_doc(3, 3, 2, DocList::FirstOfEachSplit, 9);
    store.insert(&doc).unwrap();

    // Six distinct summaries exist; only three are in the document's own list.
    assert_eq!(store.summary_count(), 6);
    let loaded = store.get_doc(3, false).unwrap().unwrap();
    assert_eq!(loaded.summaries.as_ref().unwrap().len(), 3);
    for split in &loaded.splits {
        assert_eq!(split.summaries.len(), 2);
    }

    // The three summaries that are only in split lists are still owned by the document, so a
    // delete reaches all six.
    let removed = store.delete(3).unwrap().unwrap();
    assert_eq!(removed.summary_ids.len(), 6);
    assert_eq!(store.summary_count(), 0);

    // A document with no list at all stays without one.
    let doc = build_doc(4, 2, 1, DocList::None, 3);
    store.insert(&doc).unwrap();
    assert!(store.get_doc(4, false).unwrap().unwrap().summaries.is_none());
}

#[test]
fn insert_is_rejected_before_anything_is_written() {
    let dir = TempDir::new().unwrap();
    let mut store = Store::open(dir.path(), opts()).unwrap();

    // Wrong model id (decision D5).
    let mut doc = build_doc(5, 2, 1, DocList::Union, 1);
    doc.splits[1].embedding.as_mut().unwrap().model_id = 0xBAD;
    assert!(store.insert(&doc).is_err());
    assert_eq!(store.doc_count(), 0);
    assert_eq!(store.split_count(), 0, "a rejected insert wrote nothing");

    // Wrong width.
    let mut doc = build_doc(6, 2, 1, DocList::Union, 1);
    doc.splits[0].embedding.as_mut().unwrap().embedding.truncate(3);
    assert!(store.insert(&doc).is_err());
    assert_eq!(store.doc_count(), 0);

    // Missing embedding.
    let mut doc = build_doc(7, 2, 1, DocList::Union, 1);
    doc.splits[0].summaries[0].embedding = None;
    assert!(store.insert(&doc).is_err());
    assert_eq!(store.doc_count(), 0);
}

#[test]
fn reinserting_a_url_drops_the_old_splits() {
    let dir = TempDir::new().unwrap();
    let mut store = Store::open(dir.path(), opts()).unwrap();

    let big = build_doc(8, 5, 2, DocList::Union, 2);
    store.insert(&big).unwrap();
    assert_eq!(store.split_count(), 5);
    let orphan_candidate = big.splits[4].split_id;

    // The same document id with fewer splits: the extra ones must not survive.
    let small = build_doc(8, 2, 2, DocList::Union, 77);
    let replaced = store.insert(&small).unwrap();
    assert!(replaced.existed);
    assert_eq!(store.split_count(), 2);
    assert_eq!(store.summary_count(), 4);
    assert!(store.get_split(orphan_candidate, false).unwrap().is_none());

    // Ids are deterministic, so the first two splits kept their ids and were overwritten in
    // place. Only the three that really went away are reported, or a caller removing these from
    // its index would delete vectors this same insert just wrote.
    let kept: Vec<u64> = small.splits.iter().map(|s| s.split_id).collect();
    assert_eq!(replaced.split_ids.len(), 3);
    for id in &replaced.split_ids {
        assert!(!kept.contains(id), "split {} was re-added, not removed", id);
        assert!(store.get_split(*id, false).unwrap().is_none());
    }
    let kept_summaries: Vec<u64> = small
        .splits
        .iter()
        .flat_map(|s| s.summaries.iter().map(|x| x.summary_id))
        .collect();
    assert_eq!(replaced.summary_ids.len(), 6);
    for id in &replaced.summary_ids {
        assert!(!kept_summaries.contains(id), "summary {} was re-added", id);
    }
    // And everything the new document names is readable.
    for id in kept.iter().chain(kept_summaries.iter()) {
        assert!(
            store.get_split(*id, false).unwrap().is_some()
                || store.get_summary(*id, false).unwrap().is_some()
        );
    }
}

#[test]
fn replay_reproduces_the_maps() {
    let dir = TempDir::new().unwrap();
    let seed = 0x5EED_0001;
    let mut rng = Rng::new(seed);

    let live_maps = {
        let mut store = Store::open(dir.path(), opts()).unwrap();
        let mut present: Vec<u64> = Vec::new();
        for step in 0..200u64 {
            match rng.below(10) {
                0..=5 => {
                    let doc_id = rng.below(30) + 1;
                    let n_splits = 1 + rng.below(4) as usize;
                    let list = match rng.below(3) {
                        0 => DocList::Union,
                        1 => DocList::FirstOfEachSplit,
                        _ => DocList::None,
                    };
                    let doc = build_doc(doc_id, n_splits, 2, list, step);
                    store.insert(&doc).unwrap();
                    if !present.contains(&doc_id) {
                        present.push(doc_id);
                    }
                }
                6..=8 => {
                    if !present.is_empty() {
                        let idx = rng.below(present.len() as u64) as usize;
                        let doc_id = present.remove(idx);
                        store.delete(doc_id).unwrap();
                    }
                }
                _ => {
                    store.fsync().unwrap();
                }
            }
        }
        store.fsync().unwrap();
        store.maps().clone()
    };

    let store = Store::open(dir.path(), opts()).unwrap();
    assert_eq!(
        store.maps(),
        &live_maps,
        "replay did not reproduce the live maps (seed {:#x})",
        seed
    );

    // Every location in the replayed maps resolves.
    for (id, _) in store.maps().splits.iter() {
        assert!(store.get_split(*id, true).unwrap().is_some());
    }
    for (id, _) in store.maps().summaries.iter() {
        assert!(store.get_summary(*id, true).unwrap().is_some());
    }
    for (id, _) in store.maps().docs.iter() {
        assert!(store.get_doc(*id, false).unwrap().is_some());
    }
}

#[test]
fn a_torn_tail_is_truncated_and_the_partial_insert_disappears() {
    let dir = TempDir::new().unwrap();
    let (doc_ids, segment_path) = {
        let mut store = Store::open(dir.path(), opts()).unwrap();
        for doc_id in 1..=3u64 {
            store.insert(&build_doc(doc_id, 2, 1, DocList::Union, doc_id)).unwrap();
        }
        store.fsync().unwrap();
        (vec![1u64, 2, 3], dir.path().join("records-000000.seg"))
    };

    // Cut the last few bytes: the third document's `Doc` record never completes.
    let len = std::fs::metadata(&segment_path).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(&segment_path).unwrap();
    file.set_len(len - 6).unwrap();
    drop(file);

    let store = Store::open(dir.path(), opts()).unwrap();
    assert!(store.contains_doc(doc_ids[0]));
    assert!(store.contains_doc(doc_ids[1]));
    assert!(
        !store.contains_doc(doc_ids[2]),
        "the torn insert must replay as if it never happened"
    );
    // Its splits and summaries went with it: they were buffered, never committed.
    assert_eq!(store.split_count(), 4);
    assert_eq!(store.summary_count(), 4);

    // The segment was truncated, so appends continue from a clean boundary.
    drop(store);
    let mut store = Store::open(dir.path(), opts()).unwrap();
    store.insert(&build_doc(9, 1, 1, DocList::Union, 9)).unwrap();
    drop(store);
    let store = Store::open(dir.path(), opts()).unwrap();
    assert!(store.contains_doc(9));
    assert_eq!(store.doc_count(), 3);
}

#[test]
fn segments_seal_at_the_size_threshold() {
    let dir = TempDir::new().unwrap();
    let mut options = opts();
    options.segment_max_bytes = 4 * 1024;

    let mut store = Store::open(dir.path(), options.clone()).unwrap();
    for doc_id in 1..=40u64 {
        store.insert(&build_doc(doc_id, 3, 2, DocList::Union, doc_id)).unwrap();
    }
    assert!(
        !store.manifest().sealed.is_empty(),
        "40 documents at a 4 KiB segment size must have sealed something"
    );
    let sealed = store.manifest().sealed.len();

    // Reads span sealed and active segments alike.
    for doc_id in 1..=40u64 {
        assert!(store.get_doc(doc_id, true).unwrap().is_some());
    }
    let live = store.maps().clone();
    drop(store);

    let store = Store::open(dir.path(), options).unwrap();
    assert_eq!(store.manifest().sealed.len(), sealed);
    assert_eq!(store.maps(), &live, "reopening a sealed shard changed the maps");
}

#[test]
fn compaction_keeps_every_live_record() {
    let dir = TempDir::new().unwrap();
    let mut options = opts();
    options.segment_max_bytes = 4 * 1024;

    let mut store = Store::open(dir.path(), options.clone()).unwrap();
    for doc_id in 1..=40u64 {
        store.insert(&build_doc(doc_id, 3, 2, DocList::Union, doc_id)).unwrap();
    }
    // Kill half of them, then force the active segment into the sealed set too.
    for doc_id in (1..=40u64).filter(|d| d % 2 == 0) {
        store.delete(doc_id).unwrap();
    }
    store.seal_active().unwrap();
    assert!(store.manifest().tombstone_bytes > 0);

    let before: Vec<DocumentDto> = (1..=40u64)
        .filter(|d| d % 2 == 1)
        .map(|d| store.get_doc(d, true).unwrap().unwrap())
        .collect();

    store.compact().unwrap();

    assert_eq!(store.manifest().sealed.len(), 1, "one segment after compaction");
    assert_eq!(store.manifest().tombstone_bytes, 0);
    assert_eq!(store.doc_count(), 20);

    for want in &before {
        let got = store.get_doc(want.document_id, true).unwrap().unwrap();
        assert_eq!(got.document_url, want.document_url);
        assert_eq!(got.splits.len(), want.splits.len());
        for (g, w) in got.splits.iter().zip(want.splits.iter()) {
            assert_eq!(g.split_id, w.split_id);
            assert_eq!(g.text_content, w.text_content);
            assert_eq!(g.summaries.len(), w.summaries.len());
            assert_vectors_close(
                &g.embedding.as_ref().unwrap().embedding,
                &w.embedding.as_ref().unwrap().embedding,
            );
        }
    }
    for doc_id in (1..=40u64).filter(|d| d % 2 == 0) {
        assert!(store.get_doc(doc_id, false).unwrap().is_none());
    }

    // The compacted segment replays on its own.
    let live = store.maps().clone();
    drop(store);
    let store = Store::open(dir.path(), options).unwrap();
    assert_eq!(store.maps(), &live);
}

#[test]
fn snapshot_plus_tail_replay_equals_full_replay() {
    let dir = TempDir::new().unwrap();

    let live = {
        let mut store = Store::open(dir.path(), opts()).unwrap();
        for doc_id in 1..=10u64 {
            store.insert(&build_doc(doc_id, 2, 2, DocList::Union, doc_id)).unwrap();
        }
        let seq = store.snapshot_meta().unwrap();
        assert!(seq > 0);
        // Work that lands after the snapshot and must come back from the tail.
        for doc_id in 11..=15u64 {
            store.insert(&build_doc(doc_id, 2, 2, DocList::Union, doc_id)).unwrap();
        }
        store.delete(3).unwrap();
        store.insert(&build_doc(4, 1, 1, DocList::None, 400)).unwrap();
        store.fsync().unwrap();
        store.maps().clone()
    };

    // With the snapshot: only the tail is scanned.
    let with_snapshot = Store::open(dir.path(), opts()).unwrap();
    assert!(with_snapshot.manifest().snapshot_seq > 0);
    let from_snapshot: Maps = with_snapshot.maps().clone();
    drop(with_snapshot);
    assert_eq!(from_snapshot, live);

    // Without it: the whole log is replayed. Same maps.
    meta_snapshot::remove(dir.path()).unwrap();
    let full = Store::open(dir.path(), opts()).unwrap();
    assert_eq!(
        full.maps(),
        &from_snapshot,
        "a snapshot plus tail replay must equal a full replay"
    );
}

#[test]
fn a_shard_refuses_a_different_model() {
    let dir = TempDir::new().unwrap();
    {
        let mut store = Store::open(dir.path(), opts()).unwrap();
        store.insert(&build_doc(1, 1, 1, DocList::Union, 1)).unwrap();
    }
    let other = StoreOptions::new(0xDEAD, N_EMBD);
    assert!(Store::open(dir.path(), other).is_err());

    let wrong_width = StoreOptions::new(MODEL_ID, N_EMBD * 2);
    assert!(Store::open(dir.path(), wrong_width).is_err());
}

// ---------------------------------------------------------------------------
// Regressions for defects found in review
// ---------------------------------------------------------------------------

#[test]
fn a_snapshot_that_outruns_the_log_is_discarded() {
    // `snapshot_meta` now fsyncs the log before publishing a `snapshot_seq`. This covers the
    // other half: a snapshot that somehow describes records the segment no longer holds — a lost
    // tail — must not be trusted, because replay would skip those records as already covered,
    // find no torn tail, and let the next append land on locations the maps still point at.
    let dir = TempDir::new().unwrap();
    {
        let mut store = Store::open(dir.path(), opts()).unwrap();
        for doc_id in 1..=5u64 {
            store.insert(&build_doc(doc_id, 2, 1, DocList::Union, doc_id)).unwrap();
        }
        store.snapshot_meta().unwrap();
    }

    let segment = dir.path().join("records-000000.seg");
    let len = std::fs::metadata(&segment).unwrap().len();
    let file = std::fs::OpenOptions::new().write(true).open(&segment).unwrap();
    file.set_len(len - 200).unwrap();
    drop(file);

    let mut store = Store::open(dir.path(), opts()).unwrap();
    assert!(store.doc_count() < 5, "the lost tail must not still be reported");
    // Whatever survived is readable, and appending continues from a sound boundary.
    let ids: Vec<u64> = store.maps().docs.keys().copied().collect();
    for doc_id in ids {
        assert!(store.get_doc(doc_id, true).unwrap().is_some());
    }
    store.insert(&build_doc(99, 1, 1, DocList::Union, 99)).unwrap();
    let live = store.maps().clone();
    drop(store);
    let store = Store::open(dir.path(), opts()).unwrap();
    assert_eq!(store.maps(), &live);
}

#[test]
fn dead_bytes_in_the_active_segment_reach_the_compaction_trigger() {
    // Dead records only become worth compacting once their segment is sealed, but they must not be
    // forgotten in the meantime: a replace-heavy workload that never leaves the active segment
    // would otherwise grow the log without ever tripping the ratio.
    let dir = TempDir::new().unwrap();
    let mut store = Store::open(dir.path(), opts()).unwrap();

    for doc_id in 1..=6u64 {
        store.insert(&build_doc(doc_id, 3, 2, DocList::Union, doc_id)).unwrap();
    }
    for doc_id in 1..=6u64 {
        store.insert(&build_doc(doc_id, 3, 2, DocList::Union, doc_id + 100)).unwrap();
    }
    assert_eq!(
        store.manifest().tombstone_bytes,
        0,
        "nothing is sealed yet, so nothing is compactable yet"
    );

    store.seal_active().unwrap();
    assert!(
        store.manifest().tombstone_bytes > 0,
        "sealing must hand the active segment's dead bytes to the compaction accounting"
    );
    assert!(store.compact_if_needed(0.1).unwrap(), "compaction should trigger");

    assert_eq!(store.doc_count(), 6);
    for doc_id in 1..=6u64 {
        let doc = store.get_doc(doc_id, true).unwrap().unwrap();
        assert_eq!(doc.splits.len(), 3);
    }
    assert_eq!(store.manifest().tombstone_bytes, 0);
}

#[test]
fn corruption_in_the_middle_of_the_log_is_reported_not_truncated() {
    // Only a record that runs to the end of the file can be a torn write. A bad checksum with
    // complete records after it is damage, and silently discarding everything from there would
    // throw away good documents.
    let dir = TempDir::new().unwrap();
    {
        let mut store = Store::open(dir.path(), opts()).unwrap();
        for doc_id in 1..=4u64 {
            store.insert(&build_doc(doc_id, 2, 1, DocList::Union, doc_id)).unwrap();
        }
        store.fsync().unwrap();
    }

    // Flip a bit inside the payload of the first record, far from the end of the segment.
    let segment = dir.path().join("records-000000.seg");
    let mut bytes = std::fs::read(&segment).unwrap();
    assert!(bytes.len() > 400);
    bytes[40] ^= 0xFF;
    std::fs::write(&segment, &bytes).unwrap();

    let message = match Store::open(dir.path(), opts()) {
        Ok(_) => panic!("corruption in the middle of the log was accepted"),
        Err(e) => format!("{}", e),
    };
    assert!(
        message.contains("corrupt"),
        "expected a corruption error, got: {}",
        message
    );
}

#[test]
fn a_shard_refuses_a_different_vector_dtype() {
    use embedding_store::record::VectorDtype;

    let dir = TempDir::new().unwrap();
    {
        let mut store = Store::open(dir.path(), opts()).unwrap();
        store.insert(&build_doc(1, 1, 1, DocList::Union, 1)).unwrap();
    }
    let mut other = opts();
    other.dtype = VectorDtype::F32;
    assert!(
        Store::open(dir.path(), other).is_err(),
        "an f16 shard must not be reopened as f32: the index rebuild would be handed a mix"
    );
}
