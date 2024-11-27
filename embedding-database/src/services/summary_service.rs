use anyhow::Result;
use embedding_common::prelude::*;
use std::sync::Arc;
use uuid::Uuid;
use crate::dao::summary_dao::get_summary;
use crate::prelude::*;
pub(crate) async fn save_summary(db: &Arc<RocksDB>, dto: &SummaryDto, user_id: Uuid) -> Result<()> {
    let summary = dto.to_model();
    let embedding = dto.to_embedding_model();
    let embedding_user = dto.to_embedding_user_model(user_id);
    put_summary(db, &summary).await?;
    if let Some(embedding) = embedding {
        put_embedding(db, &embedding).await?;
    }
    if let Some(embedding_user) = embedding_user {
        put_embedding_user(db, &embedding_user).await?;
    }
    Ok(())
}

pub async fn get_all_summaries_full(db: &Arc<RocksDB>, summary_ids: &Vec<u64>) -> anyhow::Result<Vec<SummaryDto>> {
    let summaries = get_all_summaries(db, summary_ids).await?;
    let summary_dtos:Vec<_> = summaries.iter().map(|summary| summary.to_dto(None)).collect();
    Ok(summary_dtos)
}

pub async fn get_summary_full(db: &Arc<RocksDB>, summary_id: &u64) -> anyhow::Result<Option<SummaryDto>> {
    let summary = get_summary(db, summary_id).await?;
    let summary_dtos = summary.map(|summary| summary.to_dto(None));
    Ok(summary_dtos)
}


