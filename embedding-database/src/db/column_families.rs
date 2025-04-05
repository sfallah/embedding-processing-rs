use strum::IntoEnumIterator;
use strum_macros::EnumIter;

#[derive(EnumIter, Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum ColumnFamilyType {
    Default,
    Documents,
    Splits,
    Summaries,
    Embeddings,
    EmbeddingFilterInfo,
    Models,
}

impl ColumnFamilyType {
    pub(crate) fn all_column_families() -> Vec<String> {
        Self::iter()
            .map(|cf| cf.name().to_string())
            .collect::<Vec<_>>()
    }
}

impl ColumnFamilyType {
    pub(crate) fn name(&self) -> &str {
        match self {
            ColumnFamilyType::Default => "default",
            ColumnFamilyType::Documents => "documents",
            ColumnFamilyType::Splits => "splits",
            ColumnFamilyType::Summaries => "summaries",
            ColumnFamilyType::Embeddings => "embeddings",
            ColumnFamilyType::EmbeddingFilterInfo => "embedding_filter_info",
            ColumnFamilyType::Models => "models",
        }
    }
}
