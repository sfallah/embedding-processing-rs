use crate::dao::document_dao::{delete_document, get_document, put_document};
use crate::db::rocksdb_impl::RocksDB;
use crate::services::split_service::save_split;
use crate::services::split_service::{delete_split_full, get_splits_full};
use crate::services::summary_service::get_summaries_full;
use anyhow::anyhow;
use embedding_common::prelude::*;
use futures::future::join_all;
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

//#[tracing::instrument(skip(db, model, embeddings, embedding_users, splits, summaries))]
pub async fn save_doc(db: &Arc<RocksDB>, dto: &DocumentDto, user_id: Uuid) -> anyhow::Result<()> {
    let document = dto.to_model();
    let split_futures = dto
        .splits
        .iter()
        .map(|split| save_split(db, split, user_id));

    join_all(split_futures)
        .await
        .into_iter()
        .collect::<anyhow::Result<()>>()?;
    put_document(db, &document).await?;
    Ok(())
}

pub async fn get_full_doc(
    db: &Arc<RocksDB>,
    document_id: u64,
    with_embeddings: bool,
    split_dtos: Option<Vec<SplitDto>>,
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
        let split_dtos = split_dtos
            .unwrap_or(get_splits_full(db, &doc.split_ids, None, None, with_embeddings).await?);
        let summary_dtos = match doc.summary_ids.clone() {
            Some(summary_ids) => {
                Some(get_summaries_full(db, &summary_ids, None, with_embeddings).await?)
            }
            None => None,
        };
        return Ok(Some(doc.to_dto(&split_dtos, summary_dtos)));
    }
    Ok(None)
}

pub async fn delete_doc_full(
    db: &Arc<RocksDB>,
    document_id: u64,
) -> anyhow::Result<Option<Document>> {
    match get_document(db, &document_id).await {
        Ok(Some(doc)) => {
            delete_document(db, &doc.document_id).await?;
            let split_ids = doc.split_ids.clone();
            let split_futures = split_ids
                .iter()
                .map(|split_id| delete_split_full(db, split_id));
            let _deleted_splits: Vec<anyhow::Result<Option<Split>>> =
                join_all(split_futures).await.into_iter().collect();
            Ok(Some(doc))
        }
        Ok(None) => Ok(None),
        Err(e) => {
            let error_message = format!("Error retrieving document: {:?}", e);
            error!("{}", error_message);
            Err(anyhow!(error_message))
        }
    }
}
