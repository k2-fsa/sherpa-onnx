fn main() {
    // Drop our display-name plugin into the generated Android project so the
    // Gradle build compiles it (the gen/ tree is created by `tauri android
    // init`; this is a no-op on other platforms and before init runs).
    let dest_dir = std::path::Path::new("gen/android/app/src/main/java/com/plugin/displayname");
    if dest_dir.parent().map(|p| p.exists()).unwrap_or(false) {
        std::fs::create_dir_all(dest_dir).ok();
        std::fs::copy(
            "android/DisplayNamePlugin.kt",
            dest_dir.join("DisplayNamePlugin.kt"),
        )
        .ok();
    }
    println!("cargo:rerun-if-changed=android/DisplayNamePlugin.kt");
    tauri_build::build()
}
