use anyhow::Result;
use embedding_common::prelude::*;
use std::sync::Arc;
use uuid::Uuid;
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
