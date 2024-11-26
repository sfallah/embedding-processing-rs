use anyhow::Result;
use embedding_common::prelude::*;
use std::sync::Arc;
use uuid::Uuid;
use crate::prelude::*;

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
