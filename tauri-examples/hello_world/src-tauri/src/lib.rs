#[tauri::command]
fn get_version() -> &'static str {
    sherpa_onnx::version()
}

#[tauri::command]
fn get_git_sha1() -> &'static str {
    sherpa_onnx::git_sha1()
}

#[tauri::command]
fn get_git_date() -> &'static str {
    sherpa_onnx::git_date()
}

#[tauri::command]
fn get_onnxruntime_version() -> &'static str {
    sherpa_onnx::onnxruntime_version()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_version,
            get_git_sha1,
            get_git_date,
            get_onnxruntime_version,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
