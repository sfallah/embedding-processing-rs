
fn main() {

    //#[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-search=/usr/local/cuda/lib64/stubs");
        println!("cargo:rustc-link-lib=cuda");
    }
    
}
