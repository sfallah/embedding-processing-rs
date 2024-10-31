use crate::dtos::document_new::DocumentDtoNew;
use crate::processing::context::ProcessingContext;
use crate::processing::splits::process_split;
use std::sync::Arc;
use crate::processing::splitter::split_text;

pub async fn process_document(
    ctx: Arc<ProcessingContext>,
    url: String,
    text: Vec<u8>,
) -> anyhow::Result<DocumentDtoNew> {
    let splits = split_text(ctx.splitter.clone(), text).await?;

    let doc_id = ctx.clone().hasher.hash(&url);
    let mut split_dtos = Vec::new();

    let split_tasks = splits.into_iter().enumerate().map(|(seq_id, split_res)| {
        let ctx_clone = ctx.clone();
        async_std::task::spawn(async move {
            process_split(
                ctx_clone,
                Arc::from(split_res.clone()),
                doc_id,
                seq_id as i32,
            )
            .await
        })
    }).collect::<Vec<_>>();

    split_tasks.into_iter().for_each(|task| {
        let split_dto = async_std::task::block_on(task).expect("Failed to process split");
        split_dtos.push(split_dto);
    });


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
