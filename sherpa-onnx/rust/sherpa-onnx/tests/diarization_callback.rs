use sherpa_onnx::OfflineSpeakerDiarization;

fn accepts_borrowed_callback(diarizer: &OfflineSpeakerDiarization) -> usize {
    let mut progress = Vec::new();
    let result = diarizer.process_with_callback(&[], |done, total| progress.push((done, total)));
    drop(result);
    progress.len()
}

#[test]
fn borrowed_callback_compiles_with_public_api() {
    let _: fn(&OfflineSpeakerDiarization) -> usize = accepts_borrowed_callback;
}
