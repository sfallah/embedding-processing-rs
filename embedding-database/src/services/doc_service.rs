use crate::dao::document_dao::{delete_document, get_document, put_document};
use crate::db::rocksdb_impl::RocksDB;
use crate::services::split_service::save_split;
use crate::services::split_service::{delete_split_full, get_splits_full};
use crate::services::summary_service::get_summaries_full;
use anyhow::anyhow;
use embedding_common::prelude::*;
use std::sync::Arc;
use tracing::error;
use uuid::Uuid;

//#[tracing::instrument(skip(db, model, embeddings, embedding_users, splits, summaries))]
pub fn save_doc(db: &Arc<RocksDB>, dto: &DocumentDto, user_id: Uuid) -> anyhow::Result<()> {
    let document = dto.to_model();
    for split in dto.splits.iter() {
        save_split(db, split, user_id)?;
    }
    put_document(db, &document)?;
    Ok(())
}

pub fn get_full_doc(
    db: &Arc<RocksDB>,
    document_id: u64,
    with_embeddings: bool,
    split_dtos: Option<Vec<SplitDto>>,
) -> anyhow::Result<Option<DocumentDto>> {
    let doc_model = match get_document(db, &document_id) {
        Ok(doc) => doc,
        Err(e) => {
            let error_message = format!("Error retrieving document: {:?}", e);
            error!("{}", error_message);
            return Err(anyhow!(error_message));
        }
    };
    if let Some(doc) = doc_model {
        let split_dtos = split_dtos.unwrap_or(get_splits_full(
            db,
            &doc.split_ids,
            None,
            None,
            with_embeddings,
        )?);
        let summary_dtos = match doc.summary_ids.clone() {
            Some(summary_ids) => Some(get_summaries_full(db, &summary_ids, None, with_embeddings)?),
            None => None,
        };
        return Ok(Some(doc.to_dto(&split_dtos, summary_dtos)));
    }
    Ok(None)
}

pub fn delete_doc_full(db: &Arc<RocksDB>, document_id: u64) -> anyhow::Result<Option<Document>> {
    match get_document(db, &document_id) {
        Ok(Some(doc)) => {
            delete_document(db, &doc.document_id)?;
            let split_ids = doc.split_ids.clone();
            for split_id in &split_ids {
                delete_split_full(db, split_id)?;
            }
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
