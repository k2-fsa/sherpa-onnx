// Doubao ASR from file: sherpa-onnx VAD + Doubao bigmodel API.
//
// Users only provide an API key and pick an audio/video file. There are no
// other settings. Results can be exported as SRT, timestamped TXT or plain
// TXT. The frontend plays the file and can seek to any transcript segment.

mod audio;
mod doubao;
mod subtitle;
mod vad;

use doubao::DoubaoClient;
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use subtitle::Segment;
use tauri::Manager;
use tauri_plugin_fs::FsExt;
#[cfg(target_os = "ios")]
use tauri_plugin_share::{ShareExt, ShareRequest};

const MODEL_FILE: &str = "silero_vad.onnx";
/// The VAD model is embedded in the binary so every build is self-contained
/// (a lone exe on Windows/Linux needs no assets/ folder). The VAD library
/// opens model *paths*, so it is extracted to the app cache on first use.
const VAD_MODEL_BYTES: &[u8] = include_bytes!("../assets/silero_vad.onnx");
const MODEL_NAME: &str = "Silero VAD v4";
const MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";
const MODEL_DOC_URL: &str = "https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html";
/// Help page describing how to get a Doubao API key.
const API_DOC_URL: &str = "https://www.doubao.com/thread/xXWsjZf94yLksVLtN";

/// Where to buy more recognition hours. Users must pick the package
/// 录音文件识别 (recording-file recognition), NOT 流式语音识别
/// (streaming speech recognition).
const PRICING_URL: &str = "https://console.volcengine.com/speech/new/purchase?chargeType=prepaid&productQueryId=10039&projectName=default";

/// Resolve display names of Android picker URIs ("msf:1495" -> "1_lei-jun.wav").
/// Everything about it is best-effort: if the Kotlin class is missing the app
/// must still start — we just fall back to the URI basename.
fn display_name_plugin<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::new("display_name")
        .setup(|app, api| {
            #[cfg(target_os = "android")]
            {
                let registered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    api.register_android_plugin("com.plugin.apphelper", "AppHelperPlugin")
                }));
                if let Ok(Ok(handle)) = registered {
                    app.manage(handle);
                }
            }
            #[cfg(not(target_os = "android"))]
            {
                let _ = (&app, &api);
            }
            Ok(())
        })
        .build()
}

/// Resets the running flag on every exit path of the worker, including when
/// it panics — otherwise one panic would block all later runs until restart.
struct RunningGuard(Arc<AtomicBool>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// Shared application state, created once at startup.
struct AppState {
    running: Arc<AtomicBool>,
    cancelled: Arc<AtomicBool>,
    progress: Arc<AtomicU32>,
    status: Arc<Mutex<String>>,
    segments: Arc<Mutex<Vec<Segment>>>,
    elapsed_secs: Arc<Mutex<f32>>,
    audio_duration_secs: Arc<Mutex<f32>>,
    /// Human-readable log lines, shown in the GUI log box.
    logs: Arc<Mutex<Vec<String>>>,
    /// How long decoding the input took, in seconds.
    decode_secs: Arc<Mutex<f32>>,
}

#[derive(Serialize, Clone)]
struct SegmentResult {
    index: u32,
    start: f32,
    end: f32,
    text: String,
}

#[derive(Serialize, Clone)]
struct ProcessingState {
    percent: u32,
    status: String,
    segments: Vec<SegmentResult>,
    elapsed_secs: f32,
    audio_duration_secs: f32,
    decode_secs: f32,
    logs: Vec<String>,
}

#[derive(Serialize, Clone)]
struct AppInfo {
    app_version: String,
    sherpa_onnx_version: String,
    git_sha1: String,
    git_date: String,
    onnxruntime_version: String,
    model_name: String,
    model_file: String,
    model_url: String,
    model_doc_url: String,
    api_doc_url: String,
    pricing_url: String,
    is_mobile: bool,
    os: String,
}

/// Return a path to the VAD model that the library can open.
///
/// The model bytes are embedded in the binary; extract them to the app cache
/// once and reuse the copy (verified by size) on later runs.
fn model_path(app: &tauri::AppHandle) -> Result<String, String> {
    let dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| e.to_string())?
        .join("models");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join(MODEL_FILE);
    let fresh = std::fs::metadata(&file)
        .map(|m| m.len() == VAD_MODEL_BYTES.len() as u64)
        .unwrap_or(false);
    if !fresh {
        std::fs::write(&file, VAD_MODEL_BYTES)
            .map_err(|e| format!("Cannot extract {MODEL_FILE}: {e}"))?;
    }
    Ok(file.display().to_string())
}

/// Output files written next to the input file, named like the Python
/// reference: `<input>.srt` (standard subtitles) and `<input>.txt`
/// (timestamped text).
fn output_paths(input_path: &str) -> (PathBuf, PathBuf) {
    let base = std::path::Path::new(input_path).with_extension("");
    (base.with_extension("srt"), base.with_extension("txt"))
}

/// Percent-decode a URI component, e.g. `%3A` -> `:`. Works on bytes so a `%`
/// next to multi-byte characters cannot panic on a UTF-8 boundary.
fn percent_decode(s: &str) -> String {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let hex = |b: u8| (b as char).to_digit(16);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// File name (with extension) of a path or URI, percent-decoded. Used to name
/// local copies of picked files and their result files.
fn input_file_name(input: &str) -> String {
    let decoded = percent_decode(input.split(['?', '#']).next().unwrap_or(input));
    let name = decoded
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or("input")
        .trim()
        .to_string();
    if name.is_empty() {
        "input".to_string()
    } else {
        name
    }
}

/// Is this a URI the picker returned (Android `content://`, iOS `file://`)
/// rather than a plain file system path?
fn is_pick_uri(input: &str) -> bool {
    input.contains("://")
}

/// Ask Android for the human-readable name of a picker URI ("msf:1495" is
/// really "1_lei-jun.wav"). Needs DisplayNamePlugin, see android/. Returns
/// None rather than ever panicking — the URI basename is the fallback.
#[cfg(target_os = "android")]
fn android_display_name(app: &tauri::AppHandle, input: &str) -> Option<String> {
    let handle = app.try_state::<tauri::plugin::PluginHandle<tauri::Wry>>()?;
    let path = input.to_string();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let v: serde_json::Value = handle
            .run_mobile_plugin("getDisplayName", serde_json::json!({ "path": path }))
            .ok()?;
        v.get("name")
            .and_then(|n| n.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }));
    result.ok().flatten()
}

/// Best-effort base name (with extension) of a picked input, used for the
/// local copy and the result file names.
#[cfg_attr(not(target_os = "android"), allow(unused_variables))]
fn input_display_name(app: &tauri::AppHandle, input: &str) -> String {
    #[cfg(target_os = "android")]
    {
        if is_pick_uri(input) {
            if let Some(name) = android_display_name(app, input) {
                if name.contains('.') {
                    return name;
                }
            }
        }
    }
    let base = input_file_name(input);
    if base.contains('.') {
        base
    } else {
        "transcript".to_string()
    }
}

/// Where the result files for a picked location go. Results cannot be written
/// beside a `content://` URI, so for picker URIs they sit next to the local
/// copy in the app cache dir.
fn resolve_outputs(app: &tauri::AppHandle, input: &str) -> Result<(PathBuf, PathBuf), String> {
    if !is_pick_uri(input) {
        // Plain desktop paths are used literally — they are not
        // percent-encoded, and names like "a%20b.wav" must survive.
        return Ok(output_paths(input));
    }
    let cache = app.path().app_cache_dir().map_err(|e| e.to_string())?;
    let name = input_display_name(app, input);
    Ok(output_paths(
        cache.join("media").join(name).to_str().unwrap_or("input"),
    ))
}

/// Return a local file path the decoder can open. Plain paths are used as-is;
/// picker URIs (`content://`, `file://`) are copied into the app cache dir via
/// the fs plugin, which knows how to read them on Android/iOS. The copy keeps
/// the original file name so tools and the media preview recognize the type.
fn materialize_input(app: &tauri::AppHandle, input: &str) -> Result<String, String> {
    if !is_pick_uri(input) {
        // Plain path — never percent-decoded (see resolve_outputs).
        if !Path::new(input).is_file() {
            return Err(format!("{input} does not exist"));
        }
        return Ok(input.to_string());
    }

    let cache = app.path().app_cache_dir().map_err(|e| e.to_string())?;
    let media_dir = cache.join("media");
    std::fs::create_dir_all(&media_dir).map_err(|e| e.to_string())?;
    let local = media_dir.join(input_display_name(app, input));

    let mut opts = tauri_plugin_fs::OpenOptions::new();
    opts.read(true);
    let fp: tauri_plugin_fs::FilePath = input
        .parse()
        .map_err(|e| format!("Cannot parse {input}: {e}"))?;
    let mut src = app
        .fs()
        .open(fp, opts)
        .map_err(|e| format!("Cannot open {input}: {e}"))?;
    let mut dst =
        std::fs::File::create(&local).map_err(|e| format!("Cannot create {}: {e}", local.display()))?;
    std::io::copy(&mut src, &mut dst).map_err(|e| format!("Cannot copy {input}: {e}"))?;
    Ok(local.display().to_string())
}

#[derive(Serialize)]
struct PreparedInput {
    local_path: String,
    base_name: String,
    size: u64,
}

// ---- result storage -------------------------------------------------------
// On Android, results go to the PUBLIC Downloads folder (via MediaStore) so
// file managers can see them. Everywhere else they go next to the input file
// (or beside the local copy for picker URIs), as before.

/// Call one of our AppHelperPlugin Kotlin commands (Android only).
#[cfg(target_os = "android")]
fn helper_call(app: &tauri::AppHandle, cmd: &str, payload: serde_json::Value) -> Result<serde_json::Value, String> {
    let handle = app
        .try_state::<tauri::plugin::PluginHandle<tauri::Wry>>()
        .ok_or_else(|| "AppHelperPlugin is unavailable".to_string())?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle.run_mobile_plugin::<serde_json::Value>(cmd, payload)
    }));
    match result {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(format!("{e}")),
        Err(_) => Err("plugin call failed".to_string()),
    }
}

/// File name of the results for an input, e.g. "1_lei-jun.srt".
fn result_name(app: &tauri::AppHandle, input: &str, ext: &str) -> String {
    let base = input_display_name(app, input);
    let stem = Path::new(&base)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "transcript".to_string());
    format!("{stem}.{ext}")
}

/// File name of a path, for display.
fn name_of(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| p.display().to_string())
}

/// Which result files already exist (returned as file names).
fn store_exists(app: &tauri::AppHandle, input: &str) -> Result<Vec<String>, String> {
    #[cfg(target_os = "android")]
    {
        if is_pick_uri(input) {
            let mut out = Vec::new();
            for ext in ["srt", "txt"] {
                let name = result_name(app, input, ext);
                let v = helper_call(app, "existsInDownloads", serde_json::json!({ "name": name }))?;
                if v.get("exists").and_then(|b| b.as_bool()).unwrap_or(false) {
                    out.push(name);
                }
            }
            return Ok(out);
        }
    }
    let (srt, txt) = resolve_outputs(app, input)?;
    Ok([srt, txt]
        .iter()
        .filter(|p| p.exists())
        .map(|p| name_of(p))
        .collect())
}

/// Read one result file, if it exists.
fn store_read(app: &tauri::AppHandle, input: &str, ext: &str) -> Result<Option<String>, String> {
    #[cfg(target_os = "android")]
    {
        if is_pick_uri(input) {
            let name = result_name(app, input, ext);
            return match helper_call(app, "readFromDownloads", serde_json::json!({ "name": name })) {
                Ok(v) => Ok(v.get("content").and_then(|c| c.as_str()).map(|s| s.to_string())),
                Err(_) => Ok(None),
            };
        }
    }
    let (srt, txt) = resolve_outputs(app, input)?;
    let p = if ext == "srt" { srt } else { txt };
    if !p.exists() {
        return Ok(None);
    }
    std::fs::read_to_string(&p)
        .map(Some)
        .map_err(|e| format!("Cannot read {}: {e}", p.display()))
}

/// Write one result file; returns its file name.
fn store_write(app: &tauri::AppHandle, input: &str, ext: &str, content: &str) -> Result<String, String> {
    #[cfg(target_os = "android")]
    {
        if is_pick_uri(input) {
            let name = result_name(app, input, ext);
            helper_call(
                app,
                "saveToDownloads",
                serde_json::json!({ "name": name, "content": content }),
            )?;
            return Ok(name);
        }
    }
    let (srt, txt) = resolve_outputs(app, input)?;
    let p = if ext == "srt" { srt } else { txt };
    std::fs::write(&p, content).map_err(|e| format!("Cannot write {}: {e}", p.display()))?;
    Ok(name_of(&p))
}

/// Delete the result files; returns the deleted file names.
fn store_delete(app: &tauri::AppHandle, input: &str) -> Result<Vec<String>, String> {
    #[cfg(target_os = "android")]
    {
        if is_pick_uri(input) {
            let mut deleted = Vec::new();
            for ext in ["srt", "txt"] {
                let name = result_name(app, input, ext);
                let v = helper_call(app, "deleteFromDownloads", serde_json::json!({ "name": name }))?;
                if v.get("deleted").and_then(|b| b.as_bool()).unwrap_or(false) {
                    deleted.push(name);
                }
            }
            return Ok(deleted);
        }
    }
    let (srt, txt) = resolve_outputs(app, input)?;
    let mut deleted = Vec::new();
    for p in [srt, txt] {
        if p.exists() {
            std::fs::remove_file(&p).map_err(|e| format!("Cannot delete {}: {e}", p.display()))?;
            deleted.push(name_of(&p));
        }
    }
    Ok(deleted)
}

/// Copy a picked input into the app cache and report the local path plus the
/// display name. The frontend uses the local path for the media preview
/// (content:// URIs cannot be played directly) and the name for outputs.
#[tauri::command]
fn prepare_input(app: tauri::AppHandle, path: String) -> Result<PreparedInput, String> {
    let local_path = materialize_input(&app, &path)?;
    let size = std::fs::metadata(&local_path).map(|m| m.len()).unwrap_or(0);
    Ok(PreparedInput {
        base_name: input_display_name(&app, &path),
        local_path,
        size,
    })
}

/// Write the two result files with plain filesystem paths. Test-only helper.
#[cfg(test)]
fn save_files(input_path: &str, segments: &[Segment]) -> Result<Vec<String>, String> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    let (srt_path, txt_path) = output_paths(input_path);
    let srt = subtitle::to_srt(segments);
    let txt = subtitle::to_timestamped_txt(segments);
    std::fs::write(&srt_path, srt)
        .map_err(|e| format!("cannot write {}: {e}", srt_path.display()))?;
    std::fs::write(&txt_path, txt)
        .map_err(|e| format!("cannot write {}: {e}", txt_path.display()))?;
    Ok(vec![
        srt_path.display().to_string(),
        txt_path.display().to_string(),
    ])
}

/// Auto-save the results. Recognition is not free, so whatever has been
/// recognized is kept even if the run was stopped early. Returns the file
/// names that were written (e.g. "1_lei-jun.srt", "Downloads/1_lei-jun.srt").
fn save_results(
    app: &tauri::AppHandle,
    input: &str,
    segments: &[Segment],
) -> Result<Vec<String>, String> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }
    #[allow(unused_mut)] // reassigned only in the Android branch below
    let mut saved = vec![
        store_write(app, input, "srt", &subtitle::to_srt(segments))?,
        store_write(app, input, "txt", &subtitle::to_timestamped_txt(segments))?,
    ];
    // Android picker inputs go to the public Downloads folder — say so.
    #[cfg(target_os = "android")]
    {
        if is_pick_uri(input) {
            saved = saved
                .into_iter()
                .map(|n| format!("Downloads/{n}"))
                .collect();
        }
    }
    Ok(saved)
}

/// Start recognition in a background thread. Returns immediately; the
/// frontend polls `get_progress` for updates.
///
/// `start_sec`/`end_sec` limit the range of the file that is sent for
/// recognition. `end_sec <= 0` (or an empty frontend field) means "to the end
/// of the file" — the default processes the whole file.
#[tauri::command]
fn recognize_file(
    app: tauri::AppHandle,
    path: String,
    api_key: String,
    start_sec: f32,
    end_sec: f32,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let api_key = api_key.trim().to_string();
    if api_key.is_empty() {
        return Err("Please provide a Doubao API key first".to_string());
    }
    // No extension gate here: picker URIs often carry no extension at all
    // (e.g. Android "msf:1495" document ids). The decoder identifies the
    // format by parsing the file content.
    let start_sec = if start_sec.is_finite() && start_sec > 0.0 {
        start_sec
    } else {
        0.0
    };
    if end_sec.is_finite() && end_sec > 0.0 && end_sec <= start_sec {
        return Err("Invalid range: the end time must be after the start time".to_string());
    }

    // Recognition is not free: refuse to run when the output files already
    // exist, before any decoding, VAD or API call — same as the Python
    // reference. Users should delete them first.
    let existing = store_exists(&app, &path)?;
    if !existing.is_empty() {
        // File names only — full paths are far too long for a phone banner.
        return Err(format!(
            "Output file(s) already exist: {}. Delete them to re-run.",
            existing.join(", ")
        ));
    }

    // Do the fallible setup BEFORE taking the running flag: if anything here
    // fails we must not leave the flag set, or every later Start would report
    // "already running" until the app restarts.
    // Plain paths are decoded in place; picker URIs (Android content://, iOS
    // file://) are copied into the app cache first so the decoder can read
    // them like a normal file.
    let local_path = materialize_input(&app, &path)?;
    let model = model_path(&app)?;

    if state.running.swap(true, Ordering::SeqCst) {
        return Err("Recognition is already running".to_string());
    }

    state.cancelled.store(false, Ordering::SeqCst);
    state.progress.store(0, Ordering::SeqCst);
    if let Ok(mut s) = state.status.lock() {
        *s = "Starting...".to_string();
    }
    if let Ok(mut s) = state.segments.lock() {
        s.clear();
    }
    if let Ok(mut l) = state.logs.lock() {
        l.clear();
        let range = if end_sec > 0.0 {
            format!("range {start_sec}s..{end_sec}s")
        } else if start_sec > 0.0 {
            format!("range {start_sec}s..end")
        } else {
            "whole file".to_string()
        };
        l.push(format!("Started: {path} ({range})"));
    }

    let running = state.running.clone();
    let cancelled = state.cancelled.clone();
    let progress = state.progress.clone();
    let status = state.status.clone();
    let segments = state.segments.clone();
    let elapsed_secs = state.elapsed_secs.clone();
    let audio_duration_secs = state.audio_duration_secs.clone();
    let logs = state.logs.clone();
    let decode_secs_out = state.decode_secs.clone();

    // The guard clears the running flag on every exit path of the worker,
    // including a panic.
    let guard = RunningGuard(running.clone());
    std::thread::spawn(move || {
        let _guard = guard;
        run_recognition(
            &local_path,
            &path,
            &app,
            &model,
            &api_key,
            start_sec,
            end_sec,
            &cancelled,
            &progress,
            &status,
            &segments,
            &elapsed_secs,
            &audio_duration_secs,
            &logs,
            &decode_secs_out,
        );
    });

    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Worker: decode, VAD, recognize each segment over HTTP, then save.
fn run_recognition(
    path: &str,
    source: &str,
    app: &tauri::AppHandle,
    model: &str,
    api_key: &str,
    start_sec: f32,
    end_sec: f32,
    cancelled: &AtomicBool,
    progress: &AtomicU32,
    status: &Mutex<String>,
    segments: &Mutex<Vec<Segment>>,
    elapsed_secs: &Mutex<f32>,
    audio_duration_secs: &Mutex<f32>,
    logs: &Mutex<Vec<String>>,
    decode_secs_out: &Mutex<f32>,
) {
    let start_time = Instant::now();
    let set_status = |s: &str| {
        if let Ok(mut st) = status.lock() {
            *st = s.to_string();
        }
    };
    let log = |msg: String| {
        if let Ok(mut l) = logs.lock() {
            l.push(msg);
        }
    };

    set_status("Decoding audio...");
    let decode_start = Instant::now();
    let samples = match audio::decode_to_mono_16k(path, start_sec, end_sec) {
        Ok(s) => s,
        Err(e) => {
            set_status(&format!("Error: {e}"));
            log(format!("Error: {e}"));
            return;
        }
    };
    let decode_secs = decode_start.elapsed().as_secs_f32();
    if let Ok(mut d) = decode_secs_out.lock() {
        *d = decode_secs;
    }
    log(format!(
        "Decoded {:.3}s of audio in {:.3}s",
        samples.len() as f32 / 16000.0,
        decode_secs
    ));
    let duration = samples.len() as f32 / 16000.0;
    if let Ok(mut d) = audio_duration_secs.lock() {
        *d = duration;
    }
    progress.store(10, Ordering::SeqCst);

    if cancelled.load(Ordering::SeqCst) {
        set_status("Cancelled");
        return;
    }

    set_status("Detecting speech...");
    let raw = match vad::run_silero_vad(&samples, model) {
        Ok(r) => r,
        Err(e) => {
            set_status(&format!("Error: {e}"));
            log(format!("Error: {e}"));
            return;
        }
    };
    progress.store(30, Ordering::SeqCst);
    log(format!("VAD detected {} speech segments", raw.len()));

    if raw.is_empty() {
        set_status("No speech detected");
        return;
    }

    set_status(&format!("Recognizing 0/{} segments...", raw.len()));
    let client = DoubaoClient::new(api_key.to_string());
    let total = raw.len();
    let done_count = Arc::new(AtomicU32::new(0));

    // Send segments to the API with a bounded pool (4 workers) and publish
    // each transcript as soon as it arrives, so the UI fills in live.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let raw = &raw;
    std::thread::scope(|scope| {
        let workers = 4.min(raw.len()).max(1);
        for _ in 0..workers {
            let next = &next;
            let raw = raw;
            let client = &client;
            let done_count = &done_count;
            let segments = segments;
            let status = status;
            let cancelled = cancelled;
            let progress = progress;
            let logs = logs;
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                if i >= raw.len() {
                    return;
                }
                if cancelled.load(Ordering::SeqCst) {
                    return;
                }
                let seg = &raw[i];
                let wav = audio::samples_to_wav(&seg.samples, 16000);
                let (text, detail) = match client.recognize(&wav) {
                    Ok((t, status)) => {
                        let t = t.trim().to_string();
                        let detail = if t.is_empty() {
                            format!("no speech: {status}")
                        } else {
                            String::new()
                        };
                        (t, detail)
                    }
                    Err(e) => {
                        eprintln!("segment {:.2}s failed: {e}", seg.start);
                        let short: String = e.chars().take(100).collect();
                        (String::new(), format!("failed: {short}"))
                    }
                };
                let n = done_count.fetch_add(1, Ordering::SeqCst) + 1;
                if let Ok(mut l) = logs.lock() {
                    // Progress only — the transcript itself shows in the
                    // result area, not here.
                    if text.is_empty() {
                        l.push(format!(
                            "[{n}/{total}] {}  ({detail})",
                            subtitle::ts_loose(start_sec + seg.start)
                        ));
                    } else {
                        l.push(format!(
                            "[{n}/{total}] {}",
                            subtitle::ts_loose(start_sec + seg.start)
                        ));
                    }
                }
                if !text.is_empty() {
                    if let Ok(mut s) = segments.lock() {
                        // Offset by the range start so times are file-absolute
                        // and click-to-play seeks to the right place.
                        s.push(Segment {
                            start: start_sec + seg.start,
                            end: start_sec + seg.end,
                            text,
                        });
                        s.sort_by(|a, b| {
                            a.start
                                .partial_cmp(&b.start)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
                progress.store(30 + (60 * n / total as u32), Ordering::SeqCst);
                if let Ok(mut st) = status.lock() {
                    *st = format!("Recognizing {n}/{total} segments...");
                }
            });
        }
    });

    // Timing stats, same fields as the Python example prints at the end:
    // audio duration, decode time, elapsed seconds and RTF.
    let elapsed = start_time.elapsed().as_secs_f32();
    let duration = audio_duration_secs.lock().map(|d| *d).unwrap_or(0.0);
    let rtf = if duration > 0.0 {
        elapsed / duration
    } else {
        0.0
    };
    if let Ok(mut e) = elapsed_secs.lock() {
        *e = elapsed;
    }
    if let Ok(mut d) = decode_secs_out.lock() {
        *d = decode_secs;
    }
    log(format!("Audio duration:\t{duration:.3} s"));
    log(format!("Decode time:\t{decode_secs:.3} s"));
    log(format!("Elapsed:\t{elapsed:.3} s"));
    log(format!("RTF = {elapsed:.3}/{duration:.3} = {rtf:.3}"));

    // Save whatever has been recognized — recognition is not free, so results
    // are written even when the run was stopped early. `source` is the
    // original input: on Android its results go to the public Downloads.
    let final_segments = segments.lock().map(|s| s.clone()).unwrap_or_default();
    match save_results(app, source, &final_segments) {
        Ok(saved) => {
            for p in &saved {
                log(format!("Saved to {p}"));
            }
            let save_msg = if saved.is_empty() {
                "nothing to save".to_string()
            } else {
                format!("Saved to {}", saved.join(" and "))
            };
            if cancelled.load(Ordering::SeqCst) {
                set_status(&format!("Stopped. {save_msg}"));
            } else {
                progress.store(100, Ordering::SeqCst);
                set_status(&format!("Done. {save_msg}"));
            }
        }
        Err(e) => {
            set_status(&format!("Error: {e}"));
            log(format!("Error: {e}"));
        }
    }
}

#[tauri::command]
/// Current progress, segments and log lines for the frontend.
fn get_progress(state: tauri::State<'_, AppState>) -> ProcessingState {
    let segments = state
        .segments
        .lock()
        .map(|s| {
            s.iter()
                .enumerate()
                .map(|(i, seg)| SegmentResult {
                    index: i as u32 + 1,
                    start: seg.start,
                    end: seg.end,
                    text: seg.text.clone(),
                })
                .collect()
        })
        .unwrap_or_default();

    ProcessingState {
        percent: state.progress.load(Ordering::Relaxed),
        status: state.status.lock().map(|s| s.clone()).unwrap_or_default(),
        segments,
        elapsed_secs: state.elapsed_secs.lock().map(|e| *e).unwrap_or(0.0),
        audio_duration_secs: state
            .audio_duration_secs
            .lock()
            .map(|d| *d)
            .unwrap_or(0.0),
        decode_secs: state.decode_secs.lock().map(|d| *d).unwrap_or(0.0),
        logs: state.logs.lock().map(|l| l.clone()).unwrap_or_default(),
    }
}

#[tauri::command]
/// Ask the running recognition to stop; results so far are kept.
fn cancel_recognition(state: tauri::State<'_, AppState>) {
    state.cancelled.store(true, Ordering::SeqCst);
}

#[tauri::command]
/// Version, model and platform information for the Info page.
fn get_info() -> AppInfo {
    AppInfo {
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        sherpa_onnx_version: sherpa_onnx::version().to_string(),
        git_sha1: sherpa_onnx::git_sha1().to_string(),
        git_date: sherpa_onnx::git_date().to_string(),
        onnxruntime_version: sherpa_onnx::onnxruntime_version().to_string(),
        model_name: MODEL_NAME.to_string(),
        model_file: MODEL_FILE.to_string(),
        model_url: MODEL_URL.to_string(),
        model_doc_url: MODEL_DOC_URL.to_string(),
        api_doc_url: API_DOC_URL.to_string(),
        pricing_url: PRICING_URL.to_string(),
        is_mobile: cfg!(any(target_os = "android", target_os = "ios")),
        os: std::env::consts::OS.to_string(),
    }
}

/// Write a result file to a user-chosen location. On Android/iOS the save
/// dialog hands back a URI (`content://`, `file://`) — those must be written
/// through the fs plugin; desktop returns plain paths.
fn write_output(app: &tauri::AppHandle, path: &str, content: &str) -> Result<(), String> {
    if is_pick_uri(path) {
        let mut opts = tauri_plugin_fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true);
        let fp: tauri_plugin_fs::FilePath = path
            .parse()
            .map_err(|e| format!("Cannot parse {path}: {e}"))?;
        let mut f = app
            .fs()
            .open(fp, opts)
            .map_err(|e| format!("Cannot write {path}: {e}"))?;
        std::io::Write::write_all(&mut f, content.as_bytes())
            .map_err(|e| format!("Cannot write {path}: {e}"))
    } else {
        // Plain path — used literally (see resolve_outputs).
        std::fs::write(path, content).map_err(|e| format!("Cannot write {path}: {e}"))
    }
}

/// Read a chunk of a local file, base64-encoded. The frontend assembles these
/// into a Blob for playback: on Android the webview's asset protocol only
/// serves APK assets, so convertFileSrc() URLs can never load files from the
/// app cache.
#[tauri::command]
fn read_media_chunk(path: String, offset: u64, len: u64) -> Result<String, String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(&path).map_err(|e| format!("Cannot open {path}: {e}"))?;
    f.seek(SeekFrom::Start(offset)).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; len.min(4 * 1024 * 1024) as usize];
    let mut filled = 0usize;
    while filled < buf.len() {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(doubao::base64_encode(&buf[..filled]))
}

/// Delete the saved result files of an input so recognition can run again.
/// Returns the deleted file names (not full paths — too long for phone UI).
#[tauri::command]
fn delete_results(app: tauri::AppHandle, path: String) -> Result<Vec<String>, String> {
    let deleted = store_delete(&app, &path)?;
    if deleted.is_empty() {
        return Err("No saved result files to delete".to_string());
    }
    Ok(deleted)
}

/// Export the transcript. `format` is one of: srt, timestamped_txt, plain_txt.
#[tauri::command]
fn export_text(
    app: tauri::AppHandle,
    path: String,
    format: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    let segments = state.segments.lock().map_err(|e| e.to_string())?.clone();
    if segments.is_empty() {
        return Err("Nothing to export yet".to_string());
    }
    let content = match format.as_str() {
        "srt" => subtitle::to_srt(&segments),
        "timestamped_txt" => subtitle::to_timestamped_txt(&segments),
        "plain_txt" => subtitle::to_plain_text(&segments),
        other => return Err(format!("Unknown export format: {other}")),
    };
    write_output(&app, &path, &content)
}

/// Share one of the result formats to another app (WeChat etc.) via the
/// system share sheet. Android/iOS only.
#[tauri::command]
fn share_result(
    app: tauri::AppHandle,
    format: String,
    name: String,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    {
        let _ = (app, format, name, state);
        Err("Sharing to other apps is available on Android and iOS".to_string())
    }
    #[cfg(any(target_os = "android", target_os = "ios"))]
    {
        let segments = state.segments.lock().map_err(|e| e.to_string())?.clone();
        if segments.is_empty() {
            return Err("Nothing to share yet".to_string());
        }
        let (content, ext) = match format.as_str() {
            "srt" => (subtitle::to_srt(&segments), "srt"),
            "timestamped_txt" => (subtitle::to_timestamped_txt(&segments), "txt"),
            "plain_txt" => (subtitle::to_plain_text(&segments), "txt"),
            other => return Err(format!("Unknown share format: {other}")),
        };

        // The share plugin takes a real file; stage it in the app cache.
        let cache = app.path().app_cache_dir().map_err(|e| e.to_string())?;
        let dir = cache.join("share");
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        let stem = input_file_name(&name);
        let stem = std::path::Path::new(&stem)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "transcript".to_string());
        let file = dir.join(format!("{stem}.{ext}"));
        std::fs::write(&file, &content)
            .map_err(|e| format!("Cannot write {}: {e}", file.display()))?;

        #[cfg(target_os = "ios")]
        {
            return app
                .share_file()
                .share_file(ShareRequest {
                    path: Some(file.display().to_string()),
                    mime: Some("text/plain".to_string()),
                    group: None,
                })
                .map(|_| ())
                .map_err(|e| format!("Share failed: {e}"));
        }

        // Android: the stock tauri-plugin-share is broken (calls share_file
        // but the Kotlin only has shareFile, and never resolves) — use our
        // AppHelperPlugin instead, which falls back to sharing plain text.
        #[cfg(target_os = "android")]
        {
            let handle = app
                .try_state::<tauri::plugin::PluginHandle<tauri::Wry>>()
                .ok_or_else(|| "AppHelperPlugin is unavailable".to_string())?;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handle.run_mobile_plugin::<serde_json::Value>(
                    "share",
                    serde_json::json!({
                        "path": file.display().to_string(),
                        "mime": "text/plain",
                        "text": content,
                    }),
                )
            }));
            return match result {
                Ok(Ok(_)) => Ok(()),
                Ok(Err(e)) => Err(format!("Share failed: {e}")),
                Err(_) => Err("Share failed".to_string()),
            };
        }
    }
}

#[derive(Serialize, Clone)]
struct LoadedResults {
    segments: Vec<SegmentResult>,
    srt_path: Option<String>,
    txt_path: Option<String>,
}

/// Load previously saved results for an input file, if any exist. Lets the
/// user review old transcripts without paying for recognition again. The
/// segments also go into the app state so export/share work on them.
#[tauri::command]
fn load_results(
    app: tauri::AppHandle,
    path: String,
    state: tauri::State<'_, AppState>,
) -> Result<Option<LoadedResults>, String> {
    let srt_content = store_read(&app, &path, "srt")?;
    let txt_content = store_read(&app, &path, "txt")?;
    let (had_srt, had_txt) = (srt_content.is_some(), txt_content.is_some());
    if !had_srt && !had_txt {
        return Ok(None);
    }

    // Prefer .srt: it has explicit cues; fall back to the timestamped txt.
    let segs = match srt_content {
        Some(content) => subtitle::parse_srt(&content)?,
        None => subtitle::parse_timestamped_txt(&txt_content.unwrap_or_default())?,
    };

    // Publish into the shared state too, so export/share work on loaded text.
    if let Ok(mut s) = state.segments.lock() {
        *s = segs.clone();
    }

    Ok(Some(LoadedResults {
        segments: segs
            .into_iter()
            .enumerate()
            .map(|(i, s)| SegmentResult {
                index: i as u32 + 1,
                start: s.start,
                end: s.end,
                text: s.text,
            })
            .collect(),
        srt_path: had_srt.then(|| result_name(&app, &path, "srt")),
        txt_path: had_txt.then(|| result_name(&app, &path, "txt")),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Tests percent-decoding of Android content URIs.
    fn percent_decode_handles_android_content_uris() {
        assert_eq!(percent_decode("raw%3A%2Fstorage%2Femulated%2F0"), "raw:/storage/emulated/0");
        assert_eq!(percent_decode("plain.txt"), "plain.txt");
    }

    #[test]
    /// Tests display-name extraction from URIs and paths.
    fn input_file_name_extracts_the_basename() {
        assert_eq!(
            input_file_name(
                "content://com.android.providers.downloads.documents/document/raw%3A%2Fstorage%2Femulated%2F0%2FDownload%2FWeiXin%2F1_lei-jun.wav"
            ),
            "1_lei-jun.wav"
        );
        assert_eq!(input_file_name("file:///a/b/my%20clip.mp4"), "my clip.mp4");
        assert_eq!(input_file_name("C:\\media\\talk.mkv"), "talk.mkv");
    }

    #[test]
    /// Tests the .srt/.txt naming used by the Python example.
    fn output_paths_follow_the_python_naming() {
        let (srt, txt) = output_paths("/a/b/lecture.mp4");
        assert_eq!(srt, PathBuf::from("/a/b/lecture.srt"));
        assert_eq!(txt, PathBuf::from("/a/b/lecture.txt"));

        let (srt, txt) = output_paths("movie.mkv");
        assert_eq!(srt, PathBuf::from("movie.srt"));
        assert_eq!(txt, PathBuf::from("movie.txt"));
    }

    #[test]
    /// Tests chunked media reads reassemble the original bytes.
    fn read_media_chunk_round_trips_a_whole_file() {
        // 2.5 chunks worth of data with a recognizable pattern.
        let dir = std::env::temp_dir().join("doubao-chunk-test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("media.bin");
        let mut data = Vec::with_capacity(2_500_000);
        for i in 0..2_500_000u32 {
            data.push((i % 251) as u8);
        }
        std::fs::write(&f, &data).unwrap();

        let chunk = 1024 * 1024u64;
        let mut out = Vec::new();
        let mut offset = 0u64;
        loop {
            let b64 = read_media_chunk(f.to_str().unwrap().to_string(), offset, chunk).unwrap();
            if b64.is_empty() {
                break;
            }
            let part = doubao::base64_decode(&b64);
            assert!(
                (part.len() as u64) <= chunk,
                "chunk larger than requested: {}",
                part.len()
            );
            out.extend_from_slice(&part);
            if (part.len() as u64) < chunk {
                break;
            }
            offset += chunk;
        }
        assert_eq!(out.len(), data.len(), "reassembled length");
        assert!(out == data, "reassembled bytes differ");
    }

    #[test]
    /// Tests auto-save contents match the export formats.
    fn save_files_writes_srt_and_timestamped_txt() {
        let dir = std::env::temp_dir().join("doubao-asr-file-test");
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("sample.wav");
        let (srt, txt) = output_paths(input.to_str().unwrap());
        let _ = std::fs::remove_file(&srt);
        let _ = std::fs::remove_file(&txt);

        let segs = vec![Segment {
            start: 1.25,
            end: 2.0,
            text: "你好".to_string(),
        }];
        let saved = save_files(input.to_str().unwrap(), &segs).unwrap();
        assert_eq!(saved.len(), 2);
        assert!(saved[0].ends_with("sample.srt"));
        assert!(saved[1].ends_with("sample.txt"));
        assert!(srt.exists());
        assert!(txt.exists());
        assert_eq!(
            std::fs::read_to_string(&srt).unwrap(),
            "1\n00:00:01,250 --> 00:00:02,000\n你好"
        );
        assert_eq!(
            std::fs::read_to_string(&txt).unwrap(),
            "0:00:01,250 --> 0:00:02,000  你好"
        );

        let _ = std::fs::remove_file(&srt);
        let _ = std::fs::remove_file(&txt);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
/// Build and start the Tauri application.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_share::init())
        // Opens URLs in the system browser — works on Android/iOS, unlike
        // tauri-plugin-shell's open() which is desktop-only.
        .plugin(tauri_plugin_opener::init())
        .plugin(display_name_plugin())
        .manage(AppState {
            running: Arc::new(AtomicBool::new(false)),
            cancelled: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(AtomicU32::new(0)),
            status: Arc::new(Mutex::new("Idle".to_string())),
            segments: Arc::new(Mutex::new(Vec::new())),
            elapsed_secs: Arc::new(Mutex::new(0.0)),
            audio_duration_secs: Arc::new(Mutex::new(0.0)),
            logs: Arc::new(Mutex::new(Vec::new())),
            decode_secs: Arc::new(Mutex::new(0.0)),
        })
        .invoke_handler(tauri::generate_handler![
            recognize_file,
            get_progress,
            cancel_recognition,
            get_info,
            export_text,
            load_results,
            share_result,
            prepare_input,
            delete_results,
            read_media_chunk,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
