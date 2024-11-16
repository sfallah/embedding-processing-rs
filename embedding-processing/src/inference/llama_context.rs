use cxx::UniquePtr;
use llama_cxx_rs::ffi;
use std::sync::Arc;

pub struct LlamaContext {
    pub ctx: Arc<UniquePtr<ffi::LlamaCxxContext>>,
}
impl LlamaContext {
    pub fn new(model_path: &str, n_batch_size: usize, n_gpu_layers: usize) -> Self {
        Self {
            ctx: Arc::new(ffi::new_llama_cxx_context(
                model_path,
                n_batch_size as i32,
                n_gpu_layers as i32,
                1,
                false,
            )),
        }
    }

    pub fn get_n_embd(&self) -> i32 {
        self.ctx.get_n_embd()
    }
    pub fn get_n_ctx(&self) -> i32 {
        self.ctx.get_n_ctx()
    }

    pub fn get_embeddings_flat(&self, texts: &Vec<String>) -> Vec<f32> {
        self.ctx.get_embeddings(&texts, false)
    }
}
fn _reshape_to_2d(flat_vec: Vec<f32>, size: usize) -> Vec<Vec<f32>> {
    let item_count = flat_vec.len() / size;
    let mut two_d_vec: Vec<Vec<f32>> = Vec::with_capacity(item_count);

    for i in 0..item_count {
        let start_index = i * size;
        let end_index = start_index + size;
        let item_vec = flat_vec[start_index..end_index].to_vec();
        two_d_vec.push(item_vec);
    }

    two_d_vec
}
