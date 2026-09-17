//! Copies a store written by the previous storage layer into per-workspace shards.
//!
//! ```shell
//! cargo run -r -p embedding-store -F rocksdb-migration --bin migrate_rocksdb -- \
//!     --from rocksdb_dir --config-file config.toml
//! ```
//!
//! The work is in `embedding_store::migrate`; this is the command line around it. It exits
//! non-zero if any document could not be migrated, having migrated the rest.

use anyhow::{Context, Result};
use clap::Parser;
use embedding_common::config::config_file::ConfigFromFile;
use embedding_common::config::AppConfig;
use embedding_common::prelude::{DeterministicAHasher, Model};
use embedding_store::migrate::{migrate, MigrationOptions};
use embedding_store::prelude::ShardOptions;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(about = "Copy a RocksDB store from the previous storage layer into per-workspace shards")]
struct Args {
    /// The RocksDB directory to read.
    #[arg(long)]
    from: String,

    /// Server config. Supplies the model and `[index]`, and the destination unless `--to` is given.
    #[arg(long, default_value = "config.toml")]
    config_file: String,

    /// Where the shards go. Defaults to the config's `[storage] dir`.
    #[arg(long)]
    to: Option<String>,

    /// Read and count everything, write nothing.
    #[arg(long)]
    dry_run: bool,

    /// Open the source read-write. The default open leaves the source untouched; use this only
    /// if a read-only open cannot see the write-ahead log, and note that it lets RocksDB recover
    /// and flush into the source directory.
    #[arg(long)]
    read_write: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let config = AppConfig::from_file(args.config_file.clone())
        .with_context(|| format!("reading {}", args.config_file))?;
    let storage = config.storage_config.clone();
    storage.validate()?;

    // Every shard carries the model id in its manifest and refuses a vector from another model.
    // The old rows carry their own model id, so a database written by a different model than the
    // config names is refused document by document, which is the point: those vectors are not
    // comparable with the ones the server would produce now.
    let hasher = DeterministicAHasher::default_hasher();
    let model_id = Model::model_id(
        &hasher,
        &config.embedding_model_info.path,
        config.embedding_model_info.n_embd,
    );

    let mut shard = ShardOptions::new(model_id, config.index_config.clone());
    shard.segment_max_bytes = storage.segment_max_bytes();

    let destination = match args.to.as_ref() {
        Some(dir) => PathBuf::from(dir),
        None => storage.storage_dir()?,
    };

    println!("source      {}", args.from);
    println!("destination {}", destination.display());
    println!(
        "model       {:#x}, {} dimensions",
        model_id, config.index_config.dimensions
    );
    if args.dry_run {
        println!("dry run, nothing will be written");
    }
    println!();

    let report = migrate(&MigrationOptions {
        source: PathBuf::from(&args.from),
        destination: destination.clone(),
        shard,
        memory_budget_mb: storage.memory_budget_mb,
        dry_run: args.dry_run,
        read_write: args.read_write,
    })?;

    println!("read {} filter rows", report.filter_rows);
    println!(
        "{} workspaces, {} distinct documents",
        report.workspaces.len(),
        report.distinct_documents
    );
    if report.shared_documents > 0 {
        println!(
            "{} documents belong to more than one workspace and are written to each",
            report.shared_documents
        );
    }
    if report.orphan_documents > 0 {
        println!(
            "{} documents have no filter row, are unreachable in the source, and are left behind",
            report.orphan_documents
        );
    }
    println!();

    for workspace in &report.workspaces {
        let written = match workspace.written {
            Some(counts) => format!("{} / {} / {}", counts.docs, counts.splits, counts.summaries),
            None => "-".to_string(),
        };
        println!(
            "{}  read {:>6} docs {:>7} splits {:>7} summaries  ->  shard {written}",
            workspace.workspace,
            workspace.read.docs,
            workspace.read.splits,
            workspace.read.summaries
        );
    }

    println!();
    println!(
        "read {} documents, {} splits, {} summaries",
        report.totals.docs, report.totals.splits, report.totals.summaries
    );
    if !args.dry_run {
        println!(
            "snapshotted {} shards under {}",
            report.shards_snapshotted,
            destination.display()
        );
    }

    if !report.failures.is_empty() {
        println!();
        println!("{} documents could not be migrated:", report.failures.len());
        for failure in report.failures.iter().take(20) {
            println!("  {failure}");
        }
        if report.failures.len() > 20 {
            println!("  ... and {} more", report.failures.len() - 20);
        }
        std::process::exit(1);
    }

    Ok(())
}
