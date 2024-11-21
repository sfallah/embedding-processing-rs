use std::sync::Arc;
use anyhow::Result;
use embedding_common::models::document::Document;
use embedding_common::models::embedding::{Embedding, EmbeddingUser};
use embedding_common::models::model::Model;
use embedding_common::models::split::Split;
use embedding_common::models::summary::Summary;
use embedding_database::dao::dao_impl::{put_document, put_embedding, put_embedding_user, put_model, put_split, put_summary};
use embedding_database::db::rocksdb_impl::RocksDB;
use futures::future::join_all;

//#[tracing::instrument(skip(db, model, embeddings, embedding_users, splits, summaries))]
pub async fn save_models_to_db(
    db: &Arc<RocksDB>,
    document: &Document,
    splits: &[Split],
    summaries: &[Summary],
    embeddings: &[Embedding],
    embedding_users: &[EmbeddingUser],
    model: Arc<Model>,
) -> Result<()> {
    put_model(db, model.clone().as_ref()).await?;

    let embedding_futures = embeddings
        .iter()
        .map(|embedding| put_embedding(db, embedding));
    join_all(embedding_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    let embedding_user_futures = embedding_users
        .iter()
        .map(|embedding_user| put_embedding_user(db, embedding_user));

    join_all(embedding_user_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    let split_futures = splits.iter().map(|split| put_split(db, split));
    join_all(split_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    let summary_futures = summaries.iter().map(|summary| put_summary(db, summary));
    join_all(summary_futures)
        .await
        .into_iter()
        .collect::<Result<()>>()?;

    put_document(db, document).await?;

    Ok(())
}
