fn main() {
    std::fs::create_dir_all("assets").expect("failed to create assets directory");
    tauri_build::build()
}
