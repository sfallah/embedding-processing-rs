#[derive(Clone, Copy, Debug)]
pub enum ColumnFamilyType {
    Default,
    Documents,
    Splits,
    Summaries,
    Embeddings,
}

impl ColumnFamilyType {
    pub(crate) fn name(&self) -> &str {
        match self {
            ColumnFamilyType::Default => "default",
            ColumnFamilyType::Documents => "documents",
            ColumnFamilyType::Splits => "splits",
            ColumnFamilyType::Summaries => "summaries",
            ColumnFamilyType::Embeddings => "embeddings",
        }
    }
}