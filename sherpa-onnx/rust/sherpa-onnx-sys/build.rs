use std::env;

fn main() {
    // Try to get library directory from environment variable
    let lib_dir = env::var("SHERPA_ONNX_LIB_DIR").ok();

    match &lib_dir {
        Some(path) => {
            println!("cargo:warning=SHERPA_ONNX_LIB_DIR={}", path);

            // Tell Rust/Cargo where to find the libraries at build time
            println!("cargo:rustc-link-search=native={}", path);

            // Add rpath for Linux/macOS
            if cfg!(any(target_os = "linux", target_os = "macos")) {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
            }
        }
        None => {
            println!("cargo:warning=SHERPA_ONNX_LIB_DIR not set. You may need to set it to the folder containing libsherpa-onnx-c-api and libonnxruntime.");
        }
    }

    // Always link against the public sherpa-onnx C API import library.
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");

    // On Unix we link directly against onnxruntime as well. The Windows release
    // archive currently ships the runtime DLLs but not an onnxruntime import
    // library, and sherpa-onnx-c-api.lib already carries that dependency.
    if !cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=dylib=onnxruntime");
    }

    // Rebuild if the env variable changes
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
}
