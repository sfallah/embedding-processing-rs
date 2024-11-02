use crate::dtos::document_new::DocumentDtoNew;
use crate::processing::context::ProcessingContext;
use crate::processing::splits::process_split;
use crate::processing::splitter::split_text;
use crate::services::embeddings::EmbeddingsRequest;
use std::sync::Arc;
use tokio::sync::mpsc;

pub async fn process_document(
    ctx: Arc<ProcessingContext>,
    embed_sender: Arc<mpsc::UnboundedSender<EmbeddingsRequest>>,
    url: String,
    text: Vec<u8>,
) -> anyhow::Result<DocumentDtoNew> {
    let splits = split_text(ctx.splitter.clone(), text).await?;

    let mut split_dtos: Vec<_> = Vec::new();
    let doc_id = ctx.clone().hasher.hash(&url);

    for (seq_id, split) in splits.iter().enumerate() {
        let split_dto = process_split(ctx.clone(), Arc::from(split.clone()), embed_sender.clone(), doc_id, seq_id as i32).await?;
        split_dtos.push(split_dto);
    }


    let summaries: Vec<_> = split_dtos
        .iter()
        .flat_map(|split| split.summaries.clone())
        .collect();

    Ok(DocumentDtoNew::new(
        doc_id,
        url.as_str(),
        split_dtos,
        summaries,
    ))
}
