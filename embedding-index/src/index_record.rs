pub struct IndexRecord {
    pub label: u64,
    pub embedding: Vec<f32>,
}

impl IndexRecord {
    pub fn new(label: u64, embedding: Vec<f32>) -> Self {
        Self { label, embedding }
    }
}
