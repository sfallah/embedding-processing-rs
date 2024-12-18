
fn main() {

    let lib_path = "/usr/local/cuda-12.2/lib64/stubs";
    // Add the library path to the linker search path
    println!("cargo:rustc-link-search=native={}", lib_path);

    // Add the library to the rpath
    println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");
    
}
