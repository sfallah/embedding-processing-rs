use std::error::Error;

#[derive(Clone, Copy)]
pub enum Table {
    Documents,
    Splits,
    Summaries,
    Embeddings,
}

impl Table {
    pub(crate) fn name(&self) -> &str {
        match self {
            Table::Documents => "documents",
            Table::Splits => "splits",
            Table::Summaries => "summaries",
            Table::Embeddings => "embeddings",
        }
    }
}

pub trait Database {
    fn open(path: &str) -> Self
    where
        Self: Sized;
    fn put(&self, table: Table, key: &u64, value: &[u8]) -> Result<(), Box<dyn Error>>;
    fn get(&self, table: Table, key: &u64) -> Result<Option<Vec<u8>>, Box<dyn Error>>;
    fn get_all(&self, table: Table) -> Result<Vec<Vec<u8>>, Box<dyn Error>>;
    fn multi_get(&self, table: Table, keys: &[u64])
        -> Result<Vec<Option<Vec<u8>>>, Box<dyn Error>>;
    fn delete(&self, table: Table, key: &u64) -> Result<(), Box<dyn Error>>;
    fn delete_many(&self, table: Table, keys: &[u64]) -> Result<(), Box<dyn Error>>;
}
