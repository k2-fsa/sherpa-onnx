use std::env;

const SHERPA_ONNX_STATIC_LIBS: &[&str] = &[
    "sherpa-onnx-c-api",
    "sherpa-onnx-core",
    "kaldi-decoder-core",
    "sherpa-onnx-kaldifst-core",
    "sherpa-onnx-fstfar",
    "sherpa-onnx-fst",
    "kaldi-native-fbank-core",
    "kissfft-float",
    "piper_phonemize",
    "espeak-ng",
    "ucd",
    "onnxruntime",
    "ssentencepiece_core",
];

fn main() {
    let lib_dir = env::var("SHERPA_ONNX_LIB_DIR").ok();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let static_link = env::var_os("CARGO_FEATURE_STATIC").is_some();

    match &lib_dir {
        Some(path) => {
            println!("cargo:warning=SHERPA_ONNX_LIB_DIR={path}");
            println!("cargo:rustc-link-search=native={path}");

            if !static_link && matches!(target_os.as_str(), "linux" | "macos") {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{path}");
            }
        }
        None => {
            println!(
                "cargo:warning=SHERPA_ONNX_LIB_DIR not set. You may need to set it to the folder containing sherpa-onnx libraries."
            );
        }
    }

    if static_link {
        emit_static_link_directives(&target_os);
    } else {
        emit_dynamic_link_directives();
    }

    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
}

fn emit_dynamic_link_directives() {
    println!("cargo:rustc-link-lib=dylib=sherpa-onnx-c-api");
    println!("cargo:rustc-link-lib=dylib=onnxruntime");
}

fn emit_static_link_directives(target_os: &str) {
    println!("cargo:warning=Static linking is enabled for sherpa-onnx");

    for lib in SHERPA_ONNX_STATIC_LIBS {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    match target_os {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        _ => {}
    }
}
