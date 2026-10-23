fn main() {
    // Drop our Android helper plugin into the generated project so the Gradle
    // build compiles it. The gen/ tree is created by `tauri android init` —
    // only when its Java source root exists is this an Android build, and
    // then the copy must succeed (no silent skips).
    let java_root = std::path::Path::new("gen/android/app/src/main/java");
    if java_root.exists() {
        let dest_dir = java_root.join("com/plugin/apphelper");
        std::fs::create_dir_all(&dest_dir)
            .expect("create gen/android apphelper package dir");
        std::fs::copy(
            "android/AppHelperPlugin.kt",
            dest_dir.join("AppHelperPlugin.kt"),
        )
        .expect("copy AppHelperPlugin.kt into the Android project");
    }
    println!("cargo:rerun-if-changed=android/AppHelperPlugin.kt");
    tauri_build::build()
}
