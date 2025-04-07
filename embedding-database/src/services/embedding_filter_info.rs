use crate::db::column_families::ColumnFamilyType;
use crate::db::db_record::DbRecordValue;
use embedding_common::prelude::{EmbeddingFilterInfo, Serde};

pub fn to_embedding_filter_info(
    embedding_user: &EmbeddingFilterInfo,
) -> anyhow::Result<DbRecordValue> {
    Ok(DbRecordValue::new(
        ColumnFamilyType::EmbeddingFilterInfo,
        embedding_user.embed_id,
        embedding_user.pack()?,
    ))
}
