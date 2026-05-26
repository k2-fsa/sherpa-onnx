#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
fn main() {
    non_streaming_speech_recognition_from_file_lib::run()
}
