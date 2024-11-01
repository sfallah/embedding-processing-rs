use embedding_processing_rs::services::embeddings::async_get_embeddings;
use embedding_processing_rs::utils::app_utils;

#[tokio::main]
async fn main() {
    run_embeddings().await.unwrap();
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
