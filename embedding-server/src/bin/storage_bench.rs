//! Storage harness for the lean-storage refactoring.
//!
//! Drives the storage layer directly, with no ZMQ and no model backend: a `ShardPool` of
//! per-workspace shards, fed a corpus and measured on insert, query and restart. Text comes from
//! the MS MARCO CSV, vectors are deterministic pseudo-random unit vectors, so what is measured is
//! storage and index latency, never retrieval relevance.
//!
//! It had a second mode, `--backend rocksdb`, that drove the old RocksDB layer over the same
//! corpus with the same vectors and the same measurements, which is the only reason results from
//! the two builds can be compared. Deleting the RocksDB layer removed this mode with it;
//! reproducing a RocksDB baseline means checking out the commit before "refactor: delete the
//! RocksDB storage layer".
//!
//! No tracing subscriber is installed, so the library's `info!` calls are dropped rather than
//! polluting the measurement.

use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::{AppConfig, IndexConfig};
use embedding_common::prelude::*;
use embedding_server::schema::search_mode::SearchModeType;
use embedding_store::prelude::{Shard, ShardOptions, ShardPool};
use indexmap::IndexMap;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Parser, Debug, Clone)]
#[command(about = "Storage benchmark: per-workspace shards")]
struct Args {
    /// Config file the index parameters and embedding dimension come from.
    #[arg(long, default_value = "config.toml")]
    config_file: String,

    /// MS MARCO CSV with a `passage` column.
    #[arg(
        long,
        default_value = "/Users/sabafallah/dev/qimia_ai_new/sources/bm25/data/msmarco_query_more_positives.csv"
    )]
    corpus: String,

    /// Directory for this run's data. Defaults to `bench_storage/shard`. Two runs must use
    /// distinct directories, and a directory that is not the server's own `[storage] dir`.
    #[arg(long)]
    dir: Option<String>,

    /// Stop once this many splits have been inserted. 0 means one pass over the corpus.
    /// Above the corpus size the corpus is cycled with perturbed document urls, so ids differ.
    #[arg(long, default_value_t = 0)]
    target_splits: usize,

    /// Passages per document, so a document has several splits.
    #[arg(long, default_value_t = 5)]
    passages_per_doc: usize,

    /// Documents are round-robined over this many workspaces.
    #[arg(long, default_value_t = 1)]
    workspaces: usize,

    /// Queries per search mode.
    #[arg(long, default_value_t = 200)]
    queries: usize,

    /// top_k per index, as the server's query handler uses it.
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Discarded queries per mode before measuring, so the first mode measured is not the only
    /// one paying a cold page cache.
    #[arg(long, default_value_t = 50)]
    warmup: usize,

    /// Keep an existing directory instead of deleting it first.
    #[arg(long)]
    keep: bool,

    /// Skip the restart measurements.
    #[arg(long)]
    skip_restart: bool,

    /// Seed for the vector generator, so a run is reproducible.
    #[arg(long, default_value_t = 0x5EED_1234_5678_9ABC)]
    seed: u64,
}

// ---------------------------------------------------------------------------
// Deterministic vectors (splitmix64; no `rand` dependency in embedding-server)
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        bits * 2.0 - 1.0
    }

    fn unit_vector(&mut self, dim: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|_| self.next_f32()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
        v
    }
}

// ---------------------------------------------------------------------------
// CSV (RFC 4180 subset: quoted fields, doubled quotes, newlines inside quotes)
// ---------------------------------------------------------------------------

fn parse_csv(content: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = content.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        field.push('"');
                    } else {
                        in_quotes = false;
                    }
                }
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' => in_quotes = true,
            ',' => row.push(std::mem::take(&mut field)),
            '\r' => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

fn load_passages(path: &str) -> anyhow::Result<Vec<String>> {
    let content = fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Failed to read corpus {}: {}", path, e))?;
    let mut rows = parse_csv(&content).into_iter();
    let header = rows
        .next()
        .ok_or_else(|| anyhow::anyhow!("Corpus {} is empty", path))?;
    let col = header
        .iter()
        .position(|h| h.trim() == "passage")
        .ok_or_else(|| anyhow::anyhow!("Corpus {} has no `passage` column", path))?;

    let mut passages = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for row in rows {
        if let Some(p) = row.get(col) {
            let p = p.trim();
            if p.len() >= 32 && seen.insert(p.to_string()) {
                passages.push(p.to_string());
            }
        }
    }
    Ok(passages)
}

// ---------------------------------------------------------------------------
// Synthetic documents shaped like the production pipeline's output
// ---------------------------------------------------------------------------

/// Sentences of at least four words, mirroring `filter_splits(.., 4)` in the pipeline.
fn sentences_of(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        current.push(c);
        if matches!(c, '.' | '!' | '?') {
            let s = current.trim().to_string();
            if s.split_whitespace().count() >= 4 {
                out.push(s);
            }
            current.clear();
        }
    }
    let s = current.trim().to_string();
    if s.split_whitespace().count() >= 4 {
        out.push(s);
    }
    out
}

fn token_len(text: &str) -> usize {
    text.split_whitespace().count()
}

/// One document: `passages` become splits, each split gets up to two summaries, and the
/// document-level summary list is the union of the split lists in split order, the shape
/// production uses.
fn build_document(
    hasher: &DeterministicAHasher,
    rng: &mut Rng,
    url: &str,
    passages: &[String],
    dim: usize,
    model_id: u64,
) -> DocumentDto {
    let doc_id = hasher.hash(&url.to_string());
    let mut split_dtos = Vec::with_capacity(passages.len());

    for (seq_id, passage) in passages.iter().enumerate() {
        let split_id = hasher.hash(&format!("{}{}", doc_id, seq_id));
        let mut summaries = Vec::new();
        for (sent_seq, sentence) in sentences_of(passage).into_iter().take(2).enumerate() {
            let no_tokens = token_len(&sentence);
            let summary_id = hasher.hash(&format!("{}{}{}", split_id, sent_seq, no_tokens));
            let embedding = EmbeddingDto::new(summary_id, rng.unit_vector(dim), model_id);
            summaries.push(SummaryDto::new(
                summary_id,
                doc_id,
                split_id,
                sent_seq as i32,
                &sentence,
                no_tokens,
                1.0 / (sent_seq as f32 + 1.0),
                Some(embedding),
                None,
                None,
            ));
        }
        let embedding = EmbeddingDto::new(split_id, rng.unit_vector(dim), model_id);
        split_dtos.push(SplitDto::new(
            split_id,
            seq_id as i32,
            doc_id,
            passage,
            token_len(passage),
            summaries,
            Some(embedding),
            None,
            None,
        ));
    }

    let doc_summaries: Vec<SummaryDto> = split_dtos
        .iter()
        .flat_map(|s| s.summaries.clone())
        .collect();

    DocumentDto::new(doc_id, url, split_dtos, Some(doc_summaries))
}

// ---------------------------------------------------------------------------
// The query path, as `embedding-server/src/api/query.rs` runs it minus rerank
// ---------------------------------------------------------------------------

/// One query against one shard: what `api/query.rs` does for a single workspace, minus the rerank
/// round trips. The workspace filter is the shard itself, so there is nothing to filter.
fn run_query_shard(
    shard: &Shard,
    doc_ids: &[u64],
    query: &[f32],
    top_k: usize,
    mode: SearchModeType,
    with_embeddings: bool,
) -> anyhow::Result<Vec<DocumentDto>> {
    let doc_filter: Option<HashSet<u64>> =
        (!doc_ids.is_empty()).then(|| doc_ids.iter().copied().collect());

    let mut split_summary_map: IndexMap<u64, Vec<SummaryDto>> = IndexMap::new();
    if mode == SearchModeType::SummaryOnly || mode == SearchModeType::SplitAndSummary {
        let hits = shard.search_summaries(query, top_k, doc_filter.as_ref())?;
        let ids: Vec<u64> = hits.keys().copied().collect();
        for summary_dto in shard.load_summaries(&ids, Some(&hits), with_embeddings)? {
            split_summary_map
                .entry(summary_dto.split_id)
                .or_default()
                .push(summary_dto);
        }
    }

    let mut split_query_res = IndexMap::new();
    if mode == SearchModeType::SplitOnly || mode == SearchModeType::SplitAndSummary {
        split_query_res = shard.search_splits(query, top_k, doc_filter.as_ref())?;
    }

    let mut splits_min_distances: IndexMap<u64, f32> =
        IndexMap::from_iter(split_summary_map.iter().map(|(k, v)| {
            let min = v
                .iter()
                .filter_map(|x| x.query_distance)
                .fold(f32::INFINITY, f32::min);
            (*k, min)
        }));
    splits_min_distances.extend(split_query_res.iter().map(|(k, v)| (*k, *v)));
    splits_min_distances.sort_by(|_, v1, _, v2| v1.partial_cmp(v2).unwrap());

    let final_split_ids: Vec<u64> = splits_min_distances.keys().copied().collect();
    let splits = shard.load_splits(
        &final_split_ids,
        Some(&split_query_res),
        Some(&split_summary_map),
        with_embeddings,
    )?;

    let mut doc_split_map: IndexMap<u64, Vec<SplitDto>> = IndexMap::new();
    for split in splits {
        doc_split_map.entry(split.doc_id).or_default().push(split);
    }

    let mut docs = Vec::new();
    for (doc_id, doc_splits) in doc_split_map.into_iter() {
        if let Some(doc) = shard.load_doc(doc_id, doc_splits, with_embeddings)? {
            docs.push(doc);
        }
    }
    Ok(docs)
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn rss_kb() -> Option<u64> {
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    for entry in entries.flatten() {
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            total += dir_size(&entry.path());
        } else {
            total += meta.len();
        }
    }
    total
}

// ---------------------------------------------------------------------------

/// What a run needs before it starts: the config, the corpus, the destination directory and the
/// workspace ids.
struct Setup {
    index_config: IndexConfig,
    dim: usize,
    hasher: DeterministicAHasher,
    model_id: u64,
    storage: embedding_common::config::StorageConfig,
    dir: PathBuf,
    passages: Vec<String>,
    target_splits: usize,
    workspace_ids: Vec<Uuid>,
}

fn setup(args: &Args) -> anyhow::Result<Setup> {
    let app_config = AppConfig::from_file(args.config_file.clone())?;
    let index_config = app_config.index_config.clone();
    let dim = index_config.dimensions;
    let n_embd = app_config.embedding_model_info.n_embd;
    if dim != n_embd as usize {
        return Err(anyhow::anyhow!(
            "[index] dimensions ({}) != [embedding_model] n_embd ({})",
            dim,
            n_embd
        ));
    }

    let hasher = DeterministicAHasher::default_hasher();
    let model_id = Model::model_id(&hasher, &app_config.embedding_model_info.path, n_embd);

    let dir = PathBuf::from(
        args.dir
            .clone()
            .unwrap_or_else(|| "bench_storage/shard".to_string()),
    );
    if dir.ends_with("storage") {
        return Err(anyhow::anyhow!(
            "refusing to use the production directory {:?}",
            dir
        ));
    }
    if dir.exists() && !args.keep {
        fs::remove_dir_all(&dir)?;
    }
    fs::create_dir_all(&dir)?;

    let passages = load_passages(&args.corpus)?;
    let target_splits = if args.target_splits == 0 {
        passages.len()
    } else {
        args.target_splits
    };

    let workspace_ids: Vec<Uuid> = (0..args.workspaces.max(1))
        .map(|i| {
            // Deterministic workspace uuids, so a rerun filters over the same ids.
            let mut bytes = [0u8; 16];
            bytes[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            bytes[8..16].copy_from_slice(&0xBE_1C_40_00_0000_0001u64.to_le_bytes());
            Uuid::from_bytes(bytes)
        })
        .collect();

    println!("# storage_bench");
    println!("corpus            : {}", args.corpus);
    println!("passages          : {}", passages.len());
    println!("target splits     : {}", target_splits);
    println!("passages/document : {}", args.passages_per_doc);
    println!("workspaces        : {}", workspace_ids.len());
    println!("dimensions        : {}", dim);
    println!(
        "index             : {} conn={} ea={} es={} {:?}",
        index_config.metric_kind,
        index_config.connectivity,
        index_config.expansion_add,
        index_config.expansion_search,
        index_config.scalar_kind
    );
    println!("data dir          : {}", dir.display());
    println!();

    Ok(Setup {
        index_config,
        dim,
        hasher,
        model_id,
        storage: app_config.storage_config,
        dir,
        passages,
        target_splits,
        workspace_ids,
    })
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let setup = setup(&args)?;
    run_shard(&args, &setup)
}

/// Delete the files a shard can rebuild, so the next open has to replay the log and rebuild both
/// graphs. `meta.snap` goes as well as `indexes.state`: with the maps gone too this is the full
/// replay: nothing is left that the log does not produce.
fn remove_derived_files(dir: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        for name in [
            "indexes.state",
            "meta.snap",
            "splits.usearch",
            "summaries.usearch",
        ] {
            let file = path.join(name);
            if file.exists() {
                fs::remove_file(&file)?;
            }
        }
    }
    Ok(())
}

/// Open the pool and load every workspace, as a cold server does on its first requests. Returns
/// how long it took to be able to answer one query.
fn open_and_warm(
    args: &Args,
    setup: &Setup,
    options: &ShardOptions,
    query: &[f32],
) -> anyhow::Result<(Duration, Arc<ShardPool>, usize, usize)> {
    let t = Instant::now();
    let pool = Arc::new(
        ShardPool::new(&setup.dir, options.clone())?
            .with_memory_budget_mb(setup.storage.memory_budget_mb),
    );
    let (mut splits, mut summaries) = (0usize, 0usize);
    for workspace in &setup.workspace_ids {
        let shard = pool.get(*workspace)?;
        let stats = shard.stats();
        splits += stats.split_index_size;
        summaries += stats.summary_index_size;
    }
    let first = pool.get(setup.workspace_ids[0])?;
    run_query_shard(
        &first,
        &[],
        query,
        args.top_k,
        SearchModeType::SplitAndSummary,
        false,
    )?;
    Ok((t.elapsed(), pool, splits, summaries))
}

fn run_shard(args: &Args, setup: &Setup) -> anyhow::Result<()> {
    let Setup {
        index_config,
        hasher,
        passages,
        workspace_ids,
        storage,
        ..
    } = setup;
    let (dim, model_id, target_splits) = (setup.dim, setup.model_id, setup.target_splits);

    let mut shard_options = ShardOptions::new(model_id, index_config.clone());
    shard_options.segment_max_bytes = storage.segment_max_bytes();
    println!("memory budget     : {} MB", storage.memory_budget_mb);
    println!("segment max       : {} MB", storage.segment_max_mb);
    println!();

    let pool = Arc::new(
        ShardPool::new(&setup.dir, shard_options.clone())?
            .with_memory_budget_mb(storage.memory_budget_mb),
    );

    // ---- insert -----------------------------------------------------------
    let mut rng = Rng::new(args.seed);
    let mut inserted_splits = 0usize;
    let mut inserted_docs = 0usize;
    let mut inserted_summaries = 0usize;
    let mut doc_seq = 0usize;
    let mut cursor = 0usize;
    let mut cycle = 0usize;

    let insert_start = Instant::now();
    while inserted_splits < target_splits {
        let mut chunk = Vec::with_capacity(args.passages_per_doc);
        while chunk.len() < args.passages_per_doc && inserted_splits + chunk.len() < target_splits {
            if cursor >= passages.len() {
                cursor = 0;
                cycle += 1;
            }
            chunk.push(passages[cursor].clone());
            cursor += 1;
        }
        if chunk.is_empty() {
            break;
        }

        let url = format!("bench://c{}/doc/{}", cycle, doc_seq);
        let workspace = workspace_ids[doc_seq % workspace_ids.len()];
        let dto = build_document(hasher, &mut rng, &url, &chunk, dim, model_id);

        // Through the pool for every document, because that is how the server reaches a shard:
        // one `get` per request, budget enforcement included.
        pool.get(workspace)?.insert(&dto)?;

        inserted_splits += dto.splits.len();
        inserted_summaries += dto.splits.iter().map(|s| s.summaries.len()).sum::<usize>();
        inserted_docs += 1;
        doc_seq += 1;

        if inserted_docs % 2000 == 0 {
            println!(
                "  .. {} docs, {} splits, {:.0} splits/s",
                inserted_docs,
                inserted_splits,
                inserted_splits as f64 / insert_start.elapsed().as_secs_f64()
            );
        }
    }
    let insert_elapsed = insert_start.elapsed();

    let (indexed_splits, indexed_summaries) =
        pool.loaded().iter().fold((0, 0), |(s, m), (_, sh)| {
            let stats = sh.stats();
            (s + stats.split_index_size, m + stats.summary_index_size)
        });

    println!();
    println!("documents         : {}", inserted_docs);
    println!("splits            : {}", inserted_splits);
    println!("summaries         : {}", inserted_summaries);
    println!(
        "vectors in index  : splits {} summaries {} (over {} loaded shard(s))",
        indexed_splits,
        indexed_summaries,
        pool.stats().loaded
    );
    println!(
        "insert            : {:.1}s, {:.0} docs/s, {:.0} splits/s",
        insert_elapsed.as_secs_f64(),
        inserted_docs as f64 / insert_elapsed.as_secs_f64(),
        inserted_splits as f64 / insert_elapsed.as_secs_f64()
    );
    println!();

    // ---- query ------------------------------------------------------------
    let mut qrng = Rng::new(args.seed ^ 0xDEAD_BEEF);
    let shard = pool.get(workspace_ids[0])?;

    for mode in [SearchModeType::SplitOnly, SearchModeType::SplitAndSummary] {
        for _ in 0..args.warmup {
            let q = qrng.unit_vector(dim);
            run_query_shard(&shard, &[], &q, args.top_k, mode, false)?;
        }

        let mut latencies = Vec::with_capacity(args.queries);
        let mut hits = 0usize;
        for _ in 0..args.queries {
            let q = qrng.unit_vector(dim);
            let t = Instant::now();
            let docs = run_query_shard(&shard, &[], &q, args.top_k, mode, false)?;
            latencies.push(t.elapsed());
            hits += docs.len();
        }
        latencies.sort();
        println!(
            "query {:<16}: p50 {:.2} ms, p99 {:.2} ms, mean {:.2} ms, {:.1} docs/query",
            mode.to_string(),
            ms(percentile(&latencies, 50.0)),
            ms(percentile(&latencies, 99.0)),
            ms(latencies.iter().sum::<Duration>() / latencies.len() as u32),
            hits as f64 / args.queries as f64
        );
    }
    drop(shard);
    println!();

    // ---- snapshot and footprint -------------------------------------------
    let log_only = dir_size(&setup.dir);
    let t = Instant::now();
    let snapshotted = pool.snapshot_all()?;
    let snapshot_elapsed = t.elapsed();
    let with_derived = dir_size(&setup.dir);

    println!(
        "snapshot          : {:.2}s for {} shard(s)",
        snapshot_elapsed.as_secs_f64(),
        snapshotted
    );
    println!(
        "RSS after load    : {:.0} MB",
        rss_kb().unwrap_or(0) as f64 / 1024.0
    );
    println!(
        "disk (log)        : {:.0} MB",
        log_only as f64 / (1024.0 * 1024.0)
    );
    println!(
        "disk (+ derived)  : {:.0} MB  <- the price of the fast restart below",
        with_derived as f64 / (1024.0 * 1024.0)
    );
    println!();

    // ---- demoted reads ----------------------------------------------------
    // What a cold shard costs. The pool demotes a shard that is over budget and opens one for a
    // read with its graphs mapped rather than resident, so this is the latency a workspace that
    // is not being written to actually gets.
    let resident_bytes = pool.stats().memory_bytes;
    let demoted = pool.evict(workspace_ids[0])?;
    let mapped_bytes = pool.stats().memory_bytes;
    if demoted {
        let shard = pool
            .peek(workspace_ids[0])
            .expect("still open, just mapped");
        println!(
            "demote            : index residency {:.0} MB -> {:.0} MB, RSS {:.0} MB",
            resident_bytes as f64 / (1024.0 * 1024.0),
            mapped_bytes as f64 / (1024.0 * 1024.0),
            rss_kb().unwrap_or(0) as f64 / 1024.0
        );

        for mode in [SearchModeType::SplitOnly, SearchModeType::SplitAndSummary] {
            for _ in 0..args.warmup {
                let q = qrng.unit_vector(dim);
                run_query_shard(&shard, &[], &q, args.top_k, mode, false)?;
            }
            let mut latencies = Vec::with_capacity(args.queries);
            for _ in 0..args.queries {
                let q = qrng.unit_vector(dim);
                let t = Instant::now();
                run_query_shard(&shard, &[], &q, args.top_k, mode, false)?;
                latencies.push(t.elapsed());
            }
            latencies.sort();
            println!(
                "mapped {:<16}: p50 {:.2} ms, p99 {:.2} ms",
                mode.to_string(),
                ms(percentile(&latencies, 50.0)),
                ms(percentile(&latencies, 99.0))
            );
        }

        // And what the first write to it pays to get the graphs back.
        let t = Instant::now();
        shard.promote()?;
        println!("promote           : {:.3}s", t.elapsed().as_secs_f64());
        println!(
            "RSS after promote : {:.0} MB",
            rss_kb().unwrap_or(0) as f64 / 1024.0
        );
    }
    println!();

    // ---- restart ----------------------------------------------------------
    if !args.skip_restart {
        let probe = qrng.unit_vector(dim);
        drop(pool);

        let (elapsed, pool, splits, summaries) =
            open_and_warm(args, setup, &shard_options, &probe)?;
        println!(
            "restart (snapshot): {:.2}s to the first query, splits {} summaries {}",
            elapsed.as_secs_f64(),
            splits,
            summaries
        );
        println!(
            "RSS after restart : {:.0} MB",
            rss_kb().unwrap_or(0) as f64 / 1024.0
        );
        drop(pool);

        remove_derived_files(&setup.dir)?;
        let (elapsed, _pool, splits, summaries) =
            open_and_warm(args, setup, &shard_options, &probe)?;
        println!(
            "restart (replay)  : {:.2}s to the first query, splits {} summaries {}",
            elapsed.as_secs_f64(),
            splits,
            summaries
        );
        println!(
            "RSS after replay  : {:.0} MB",
            rss_kb().unwrap_or(0) as f64 / 1024.0
        );
    }

    Ok(())
}
