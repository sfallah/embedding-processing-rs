use crate::dao::document_dao::get_splits_of_document;
use crate::dao::split_dao::get_all_splits_full;
use crate::prelude::*;
use crate::services::split_service::save_split;
use crate::services::summary_service::{get_all_summaries_full, save_summary};
use anyhow::anyhow;
use embedding_common::prelude::*;
use futures::future::join_all;
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

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

pub async fn get_full_doc(
    db: &Arc<RocksDB>,
    document_id: u64,
) -> anyhow::Result<Option<DocumentDto>> {
    let doc_model = match get_document(db, &document_id).await {
        Ok(doc) => doc,
        Err(e) => {
            let error_message = format!("Error retrieving document: {:?}", e);
            error!("{}", error_message);
            return Err(anyhow!(error_message));
        }
    };
    if let Some(doc) = doc_model {
        let split_dtos = get_all_splits_full(db, &doc.split_ids).await?;
        let summary_ids = doc.summary_ids.clone().unwrap_or_default();
        let summary_dtos = get_all_summaries_full(db, &summary_ids).await?;
        //FIXME: Order the summaries by their order in the document
        return Ok(Some(doc.to_dto(&split_dtos, &summary_dtos)));
    }
    Ok(None)
}
