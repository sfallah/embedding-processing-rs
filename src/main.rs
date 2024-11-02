use embedding_processing_rs::processing::documents::process_document;
use embedding_processing_rs::services::embeddings::async_get_embeddings;
use embedding_processing_rs::utils::app_utils;
use embedding_processing_rs::utils::app_utils::init_ctx;

#[tokio::main]
async fn main() {
    //run_embeddings().await.unwrap();
    run_doc_processing().await.unwrap();
}

async fn run_doc_processing() -> anyhow::Result<()> {
    let (embed, shutdown,handle) = app_utils::init().await?;
    let ctx = init_ctx().await;
    let file = "tests/test_data/superlinear.txt";
    let doc = tokio::fs::read_to_string(file).await?;
    let res = process_document(ctx.clone(), embed.clone(), file.to_string(), doc.into_bytes().to_vec()).await?;
    println!("{:?}", res);
    shutdown.send("shutdown".to_string())?;
    handle.await?;
    Ok(())
}

async fn run_embeddings() -> anyhow::Result<()> {
    let (embed, shutdown,handle) = app_utils::init().await?;
    let text = "This is a test".to_string();
    let embeddings = async_get_embeddings(embed.clone(), &vec![text.clone()], 512).await?;
    println!("{:?}", embeddings);
    let embeddings = async_get_embeddings(embed.clone(), &vec![text], 512).await?;
    println!("got second embeddings");
    shutdown.send("shutdown".to_string())?;
    //tokio::time::sleep(std::time::Duration::from_secs(10)).await;
    handle.await?;
    Ok(())
}
