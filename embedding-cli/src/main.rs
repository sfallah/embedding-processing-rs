use clap::Parser;
use futures::future::join_all;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, error, info, trace};

use anyhow::Result;
use async_channel::Sender;
use embedding_common::utils::helpers::get_db_dir;

use embedding_processing::processing::documents::process_document;
use embedding_processing::utils::app_utils;

use embedding_processing::utils::app_utils::{init_ctx, setup_tracing};

use embedding_cli::Cli;
use embedding_common::config::AppConfig;
use embedding_common::prelude::{DocumentDto, Model};
use embedding_index::hnsw_index::HnswIndex;
use embedding_processing::processing::context::ProcessingContext;
use embedding_processing::services::embeddings::{async_get_embeddings, EmbeddingsRequest};
use uuid::Uuid;
use embedding_database::prelude::*;
use embedding_index::{add_to_indices, save_index};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();
    setup_tracing(args.log_level);
    debug!("{:?}", args);

    if let Err(e) = args.validate() {
        error!("Invalid arguments: {}", e);
        std::process::exit(1);
    }
    info!("Starting up");

    let app_config = AppConfig::from_file_async(args.config_file).await?;

    let db_config = app_config.clone().database_config;

    let db_path_binding = get_db_dir(Some(&db_config.rocksdb_dir))?;
    let db_path = db_path_binding.to_str().unwrap();
    embedding_common::utils::helpers::create_directory(db_path)
        .expect("Failed to create directory");
    let rocksdb = RocksDB::open(&db_path).await?;
    let db = Arc::new(rocksdb);

    let user_id = args
        .user_id
        .map_or(Uuid::new_v4(), |user_id| Uuid::parse_str(&user_id).unwrap());

    info!("User ID: {}", user_id);
    match args.command {
        embedding_cli::Commands::Query { query, top_k } => {
            info!("Querying");
            run_query(db, user_id, app_config.clone(), query, top_k).await?;
        }
        embedding_cli::Commands::Index { ref file_path } => {
            run_process_docs(db, file_path, user_id, app_config.clone()).await?;
        }
    }
    info!("Shutting down");
    Ok(())
}

#[tracing::instrument(skip(db, app_config))]
async fn run_query(
    db: Arc<RocksDB>,
    user_id: Uuid,
    app_config: AppConfig,
    query: String,
    top_k: usize,
) -> Result<()> {
    let model_config = app_config.model_config;

    let splits_index =
        HnswIndex::async_create_index("splits".to_string(), app_config.index_config.clone()).await?;
    let splits_index = Arc::new(splits_index);

    //let summaries_index = HnswIndex::load_index("summaries".to_string(), app_config.index_config.clone())?;

    let (embed, shutdown, handles, model) =
        app_utils::init(model_config.clone()).await?;

    let query_embd = async_get_embeddings(
        embed.clone(),
        &vec![query.to_string()],
        model.n_embd as usize,
    )
    .await?;

    trace!("Query embeddings: {:?}", query_embd.len());

    let res = splits_index
        .query_filter(&db, &vec![user_id.clone()], &query_embd.to_vec(), top_k)
        .await?;
    let embd_ids:Vec<_> = res.keys().map(|x|*x).collect();
    info!("embd_ids: {:?}", embd_ids);

    let splits = get_splits_full(&db, &embd_ids, None, None, false).await?;
    for split in splits {
        info!("Split: {:?}", split);
    }

    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;
    Ok(())
}

//#[tracing::instrument(skip(db,app_config))]
async fn run_process_docs(
    db: Arc<RocksDB>,
    file_path: &std::path::Path,
    user_id: Uuid,
    app_config: AppConfig,
) -> Result<()> {
    let model_config = app_config.model_config;
    let splitter_config = app_config.splitter_config;

    let splits_index =
        HnswIndex::async_create_index("splits".to_string(), app_config.index_config.clone()).await?;
    let splits_index = Arc::new(splits_index);
    let summaries_index =
        HnswIndex::async_create_index("summaries".to_string(), app_config.index_config.clone()).await?;
    let summaries_index = Arc::new(summaries_index);

    let text_files =
        embedding_cli::file_io::list_files(&file_path, &vec!["txt".to_string()]).await?;

    let (embed, shutdown, handles, model) =
        app_utils::init(model_config.clone()).await?;

    let ctx = init_ctx(
        splitter_config.max_tokens,
        splitter_config.merge_level,
        model.n_embd as usize,
        model.model_id,
    )
    .await;

    let futures = text_files.iter().map(|file_path| {
        process_doc(
            db.clone(),
            splits_index.clone(),
            summaries_index.clone(),
            ctx.clone(),
            embed.clone(),
            file_path,
            user_id,
            model.clone(),
        )
    });
    join_all(futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;
    save_index(splits_index, summaries_index).await?;
    shutdown.send("shutdown".to_string())?;
    futures::future::join_all(handles.into_iter()).await;
    Ok(())
}

//#[tracing::instrument(skip(db, splits_index, summaries_index, ctx, embed))]
async fn process_doc(
    db: Arc<RocksDB>,
    splits_index: Arc<HnswIndex>,
    summaries_index: Arc<HnswIndex>,
    ctx: Arc<ProcessingContext>,
    embed: Arc<Sender<EmbeddingsRequest>>,
    file_path: &PathBuf,
    user_id: Uuid,
    model: Arc<Model>,
) -> Result<()> {
    let doc = tokio::fs::read_to_string(file_path).await?;

    let doc_dto = process_document(
        ctx.clone(),
        embed.clone(),
        file_path.to_string_lossy().to_string(),
        doc.into_bytes().to_vec(),
    )
    .await?;

    println!("{:?}", doc_dto);

    save_to_db(&db, &doc_dto, model.clone(), user_id).await?;
    add_to_indices(splits_index, summaries_index, &doc_dto).await
}



//#[tracing::instrument(skip(db, model, embeddings, embedding_users, splits, summaries))]
async fn save_to_db(
    db: &Arc<RocksDB>,
    dto: &DocumentDto,
    _model: Arc<Model>,
    user_id: Uuid,
) -> Result<()> {
    //put_model(db, model.clone().as_ref()).await?;
    save_doc(db, dto, user_id).await?;
    Ok(())
}
