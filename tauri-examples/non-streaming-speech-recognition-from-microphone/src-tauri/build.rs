fn main() {
    let _ = std::fs::create_dir_all("assets");
    tauri_build::build()
}
