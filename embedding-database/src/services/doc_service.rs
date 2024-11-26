use embedding_common::prelude::*;
use futures::future::join_all;
use std::sync::Arc;
use uuid::Uuid;
use crate::prelude::*;
use crate::services::split_service::save_split;
use crate::services::summary_service::save_summary;

//#[tracing::instrument(skip(db, model, embeddings, embedding_users, splits, summaries))]
pub async fn save_doc(db: &Arc<RocksDB>, dto: &DocumentDto, user_id: Uuid) -> anyhow::Result<()> {
    let document = dto.to_model();
    put_document(db, &document).await?;

    let split_futures = dto
        .splits
        .iter()
        .map(|split| save_split(db, split, user_id));
    join_all(split_futures)
        .await
        .into_iter()
        .collect::<anyhow::Result<()>>()?;

    let summary_futures = dto
        .summaries
        .iter()
        .map(|summary| save_summary(db, summary, user_id));
    join_all(summary_futures)
        .await
        .into_iter()
        .collect::<anyhow::Result<()>>()?;

    Ok(())
}
