#[derive(Clone, Copy)]
pub enum ColumnFamilyType {
    Documents,
    Splits,
    Summaries,
    Embeddings,
}

impl ColumnFamilyType {
    pub(crate) fn name(&self) -> &str {
        match self {
            ColumnFamilyType::Documents => "documents",
            ColumnFamilyType::Splits => "splits",
            ColumnFamilyType::Summaries => "summaries",
            ColumnFamilyType::Embeddings => "embeddings",
        }
    }
}