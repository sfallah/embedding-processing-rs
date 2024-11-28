use zeromq::{DealerSocket, RouterSocket, Socket};

pub struct ServerTask {
    pub frontend: RouterSocket,
    pub backend: DealerSocket,
}

impl ServerTask {
    pub async fn init(host: &str, frontend_port: usize, backend_port: usize) -> Self {
        let mut frontend = RouterSocket::new();
        let frontend_endpoint = format!("tcp://{}:{}", host, frontend_port);
        frontend
            .bind(&frontend_endpoint)
            .await
            .expect("Server failed binding frontend");

        let mut backend = DealerSocket::new();
        let backend_endpoint = format!("tcp://{}:{}", host, backend_port);
        backend
            .bind(&backend_endpoint)
            .await
            .expect("Server failed binding backend");

        ServerTask { frontend, backend }
    }
}
