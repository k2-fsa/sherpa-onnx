fn main() {
    // Drop our Android helper plugin into the generated project so the Gradle
    // build compiles it (the gen/ tree is created by `tauri android init`;
    // this is a no-op on other platforms and before init runs).
    let dest_dir = std::path::Path::new("gen/android/app/src/main/java/com/plugin/apphelper");
    if dest_dir.parent().map(|p| p.exists()).unwrap_or(false) {
        std::fs::create_dir_all(dest_dir).ok();
        std::fs::copy(
            "android/AppHelperPlugin.kt",
            dest_dir.join("AppHelperPlugin.kt"),
        )
        .ok();
    }
    println!("cargo:rerun-if-changed=android/AppHelperPlugin.kt");
    tauri_build::build()
}
