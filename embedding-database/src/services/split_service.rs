use crate::prelude::*;
use anyhow::{anyhow, Result};
use embedding_common::prelude::*;
use std::sync::Arc;
use uuid::Uuid;

pub(crate) async fn save_split(
    db: &Arc<RocksDB>,
    split_dto: &SplitDto,
    user_id: Uuid,
) -> Result<()> {
    let split = split_dto.to_model();
    let embedding = split_dto.to_embedding_model();
    let embedding_user = split_dto.to_embedding_user_model(user_id);
    put_split(db, &split).await?;
    if let Some(embedding) = embedding {
        put_embedding(db, &embedding).await?;
    }
    if let Some(embedding_user) = embedding_user {
        put_embedding_user(db, &embedding_user).await?;
    }
    Ok(())
}

pub async fn get_split_full(db: &Arc<RocksDB>, split_id: u64) -> Result<Option<SplitDto>> {
    match get_split(&db, &split_id).await? {
        Some(split) => {
            let summary_ids = split.summary_ids.clone().unwrap_or_else(|| Vec::new());
            let summaries = get_all_summaries(&db, &summary_ids).await?;
            let summary_dtos: Vec<_> = summaries.iter().map(|s| s.to_dto(None)).collect();
            let split_dto = split.to_dto_full(summary_dtos);
            Ok(Some(split_dto))
        }
        None => return Ok(None),
    }
}
