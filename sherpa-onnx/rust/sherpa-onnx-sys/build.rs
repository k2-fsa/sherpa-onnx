use std::env;

fn main() {
    // Try to get library directory from environment variable
    let lib_dir = env::var("SHERPA_ONNX_LIB_DIR").ok();

    match &lib_dir {
        Some(path) => {
            println!("cargo:warning=SHERPA_ONNX_LIB_DIR={}", path);

            // Tell Rust/Cargo where to find the libraries at build time
            println!("cargo:rustc-link-search=native={}", path);

            // Add rpath so the dynamic linker can find the libraries at runtime
            if cfg!(target_os = "linux") {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
            } else if cfg!(target_os = "macos") {
                // Use absolute path from env
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
            }
        }
        None => {
            println!("cargo:warning=SHERPA_ONNX_LIB_DIR not set. You may need to set it to the folder containing libsherpa-onnx-c-api and libonnxruntime.");
        }
    }

    // Link the dynamic libraries regardless (cargo will fail later if not found)
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
    println!("cargo:rustc-link-lib=dylib=onnxruntime");

    // Rebuild if the env variable changes
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
}
