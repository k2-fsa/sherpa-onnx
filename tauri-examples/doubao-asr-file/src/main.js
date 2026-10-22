const { invoke, convertFileSrc } = window.__TAURI__.core;
const { open, save, confirm } = window.__TAURI__.dialog;
const { open: openUrl } = window.__TAURI__.shell;

const AUDIO_EXT = [
  "wav", "mp3", "flac", "ogg", "oga", "m4a", "aac", "opus", "weba", "wma", "aiff", "aif",
];
const VIDEO_EXT = ["mp4", "webm", "mov", "mkv", "avi", "m4v", "ogv", "3gp"];

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- tabs
function switchTab(name) {
  for (const t of ["asr", "info"]) {
    $(`tab-${t}`).classList.toggle("active", t === name);
    $(`nav-${t}`).classList.toggle("active", t === name);
  }
}
$("nav-asr").addEventListener("click", () => switchTab("asr"));
$("nav-info").addEventListener("click", () => switchTab("info"));

// ---------------------------------------------------------------- api key
const KEY_STORAGE = "doubao_api_key";
const apiKeyInput = $("api-key");
apiKeyInput.value = localStorage.getItem(KEY_STORAGE) || "";
apiKeyInput.addEventListener("input", () => {
  localStorage.setItem(KEY_STORAGE, apiKeyInput.value.trim());
});
$("toggle-key").addEventListener("click", () => {
  const showing = apiKeyInput.type === "text";
  apiKeyInput.type = showing ? "password" : "text";
  $("toggle-key").textContent = showing ? "Show" : "Hide";
});

// Show/hide the info block (key help, free hours, privacy) — shown by default.
$("toggle-info").addEventListener("click", () => {
  const hidden = $("key-info").classList.toggle("hidden");
  $("toggle-info").textContent = hidden ? "Show info" : "Hide info";
});

// Clear the stored API key — confirm first, since it is not free to lose
// track of where the key lives.
$("clear-key").addEventListener("click", async () => {
  const ok = await confirm(
    "This will delete the API key stored on this computer.\n\n" +
      "You will need to paste the key again before the next recognition. " +
      "The key itself is not deleted from your Volcengine account.\n\n" +
      "Continue?",
    { title: "Clear API key", type: "warning" },
  );
  if (!ok) return;
  localStorage.removeItem(KEY_STORAGE);
  apiKeyInput.value = "";
  showError("");
  $("status").textContent = "API key cleared from this computer.";
  appendLog("API key cleared from this computer.");
});

function showError(msg) {
  const el = $("error");
  if (!msg) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = msg;
  el.classList.remove("hidden");
}

// ---------------------------------------------------------------- links
function wireLink(id, url) {
  const el = $(id);
  if (!el) return;
  el.href = url;
  el.addEventListener("click", (e) => {
    e.preventDefault();
    openUrl(url);
  });
}

// ---------------------------------------------------------------- media
let selectedPath = null;

/** Recent input files, newest first, kept across app restarts. */
const RECENT_KEY = "recent_input_files";
const RECENT_MAX = 50;

function loadRecent() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
  } catch {
    return [];
  }
}

function rememberFile(path) {
  const list = loadRecent().filter((p) => p !== path);
  list.unshift(path);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, RECENT_MAX)));
  renderRecent();
}

function renderRecent() {
  const list = loadRecent();
  const wrap = $("recent-wrap");
  const box = $("recent-list");
  box.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "Select a recent file…";
  box.appendChild(placeholder);
  wrap.classList.toggle("hidden", list.length === 0);
  for (const path of list) {
    const opt = document.createElement("option");
    opt.value = path;
    opt.textContent = path;
    box.appendChild(opt);
  }
}

$("recent-list").addEventListener("change", (e) => {
  const path = e.target.value;
  if (path) selectFile(path);
});

$("clear-history").addEventListener("click", () => {
  localStorage.removeItem(RECENT_KEY);
  renderRecent();
  $("status").textContent = "Recent files history cleared.";
  appendLog("Recent files history cleared.");
});

/** Select an input file (from the dialog or from the recent list). */
async function selectFile(path) {
  selectedPath = path;
  $("file-name").textContent = path.split(/[\\/]/).pop();
  showError("");

  const media = $("media");
  media.src = convertFileSrc(path);
  // The <video> element plays audio-only files as well (blank frame).
  $("media-wrap").classList.remove("hidden");
  resetResult();
  rememberFile(path);
  appendLog(`Selected: ${path}`);
  await tryLoadExisting(path);
}

$("media").addEventListener("error", () => {
  if (selectedPath) {
    showError(`Cannot open ${selectedPath} — has the file moved or been deleted?`);
  }
});

$("pick-file").addEventListener("click", async () => {
  const selected = await open({
    multiple: false,
    filters: [{ name: "Audio & video", extensions: [...AUDIO_EXT, ...VIDEO_EXT] }],
  });
  if (!selected) return;
  await selectFile(selected);
});

renderRecent();

/** Append a client-side line to the log box. */
function appendLog(msg) {
  const box = $("log-box");
  const line = document.createElement("div");
  line.className = "log-line";
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}

/**
 * If <name>.srt / <name>.txt already exist next to the input, load them and
 * show the old transcript — recognition is not free, so never redo paid work.
 * Start still refuses to run until those files are deleted or renamed.
 */
async function tryLoadExisting(selected) {
  try {
    const loaded = await invoke("load_results", { path: selected });
    if (!loaded || !loaded.segments || loaded.segments.length === 0) return;
    segments = loaded.segments.map((s) => ({
      index: s.index,
      start: s.start,
      end: s.end,
      text: s.text,
    }));
    recomputeMatches();
    renderResult();
    updateExportButtons();
    const src = loaded.srt_path || loaded.txt_path;
    $("status").textContent = `Loaded existing results from ${src}`;
    appendLog(`Loaded existing results: ${src} (${segments.length} segments)`);
    appendLog("Delete or rename the .srt/.txt files before pressing Start to re-run.");
  } catch (e) {
    console.error("load_results failed", e);
  }
}

/** Seek the media preview to `sec` and play — used when a row is clicked. */
function seekAndPlay(sec) {
  const media = $("media");
  if (!media.src) return;
  media.currentTime = sec;
  media.play().catch(() => {
    /* autoplay may be blocked until the user interacts; ignore */
  });
}

// ---------------------------------------------------------------- subtitles
/** Subtitles are shown over the player by default (video and audio alike). */
let subtitlesOn = true;

$("toggle-subtitles").addEventListener("click", () => {
  subtitlesOn = !subtitlesOn;
  $("toggle-subtitles").textContent = subtitlesOn ? "Subtitles: On" : "Subtitles: Off";
  updateSubtitleOverlay();
});

function updateSubtitleOverlay() {
  const overlay = $("subtitle-overlay");
  const media = $("media");
  const t = media ? media.currentTime : 0;
  const seg = segments.find((s) => t >= s.start && t < s.end);
  const text = seg ? seg.text : "";
  $("subtitle-text").textContent = text;
  overlay.classList.toggle("hidden", !subtitlesOn || !text);
}

$("media").addEventListener("timeupdate", updateSubtitleOverlay);
$("media").addEventListener("seeked", updateSubtitleOverlay);

// ---------------------------------------------------------------- state
/** Latest transcript segments, in file-absolute seconds. */
let segments = [];
/** Which format the result area is showing. */
let viewFormat = "timestamped";
/** Search state. */
let searchQuery = "";
let matches = []; // indices into `segments`
let currentHit = -1;

function resetResult() {
  segments = [];
  searchQuery = "";
  matches = [];
  currentHit = -1;
  logShown = 0;
  $("search").value = "";
  $("search-count").textContent = "";
  $("log-box").innerHTML = "";
  $("stats").textContent = "";
  renderResult();
  $("progress-bar").style.width = "0%";
  updateExportButtons();
}

function updateExportButtons() {
  // Keep everything that was recognized so far, including after Stop.
  const enabled = segments.length > 0;
  for (const id of [
    "export-srt",
    "export-timestamped",
    "export-plain",
    "share-srt",
    "share-timestamped",
    "share-plain",
  ]) {
    // Null-safe: a missing button must never break the render loop.
    const el = $(id);
    if (el) el.disabled = !enabled;
  }
}

function formatClock(t) {
  const s = Math.max(0, t);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.floor(s % 60);
  const ms = Math.round((s - Math.floor(s)) * 1000);
  return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")},${String(
    Math.min(999, ms),
  ).padStart(3, "0")}`;
}

function formatSrtTime(t) {
  return formatClock(t).padStart(12, "0");
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Wrap every search hit in <mark>; input is escaped first. */
function highlight(text, query) {
  const safe = escapeHtml(text);
  const q = query ? escapeHtml(query) : "";
  if (!q) return safe;
  const needle = q.toLowerCase();
  const lower = safe.toLowerCase();
  let out = "";
  let pos = 0;
  while (true) {
    const i = lower.indexOf(needle, pos);
    if (i < 0) break;
    out += safe.slice(pos, i) + "<mark>" + safe.slice(i, i + q.length) + "</mark>";
    pos = i + q.length;
  }
  return out + safe.slice(pos);
}

// ---------------------------------------------------------------- search
function recomputeMatches() {
  const q = searchQuery.trim().toLowerCase();
  matches = [];
  if (q) {
    segments.forEach((seg, i) => {
      if (seg.text.toLowerCase().includes(q)) matches.push(i);
    });
  }
  currentHit = matches.length > 0 ? 0 : -1;
}

function renderSearchCount() {
  const countEl = $("search-count");
  if (!searchQuery.trim()) {
    countEl.textContent = "";
  } else if (matches.length === 0) {
    countEl.textContent = "no match";
  } else {
    countEl.textContent = `${currentHit + 1}/${matches.length}`;
  }
  $("search-prev").disabled = matches.length === 0;
  $("search-next").disabled = matches.length === 0;
}

function gotoHit(delta) {
  if (matches.length === 0) return;
  currentHit = (currentHit + delta + matches.length) % matches.length;
  renderResult();
  const row = document.querySelector(".segment-row.active");
  if (row) row.scrollIntoView({ block: "center", behavior: "smooth" });
}

$("search").addEventListener("input", (e) => {
  searchQuery = e.target.value;
  recomputeMatches();
  renderResult();
});
$("search").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    gotoHit(e.shiftKey ? -1 : 1);
  } else if (e.key === "Escape") {
    $("search").value = "";
    searchQuery = "";
    recomputeMatches();
    renderResult();
  }
});
$("search-prev").addEventListener("click", () => gotoHit(-1));
$("search-next").addEventListener("click", () => gotoHit(1));

// ---------------------------------------------------------------- render
for (const btn of document.querySelectorAll("#format-tabs .fmt")) {
  btn.addEventListener("click", () => {
    viewFormat = btn.dataset.format;
    for (const b of document.querySelectorAll("#format-tabs .fmt")) {
      b.classList.toggle("active", b === btn);
    }
    renderResult();
  });
}

function hitClass(i) {
  if (matches.length > 0 && matches[currentHit] === i) return "active";
  if (matches.includes(i)) return "hit";
  return "";
}

function renderResult() {
  const wrap = $("segments");
  wrap.innerHTML = "";
  if (segments.length === 0) {
    const empty = document.createElement("p");
    empty.className = "hint";
    empty.textContent = "No transcript yet.";
    wrap.appendChild(empty);
    renderSearchCount();
    return;
  }

  const q = searchQuery.trim();
  if (viewFormat === "plain") {
    // One paragraph of text; still clickable per segment.
    const p = document.createElement("p");
    p.className = "plain-text";
    p.innerHTML = segments
      .map(
        (seg, i) =>
          `<span class="plain-seg ${hitClass(i)}" data-i="${i}">${highlight(seg.text, q)}</span>`,
      )
      .join("");
    wrap.appendChild(p);
    for (const span of wrap.querySelectorAll(".plain-seg")) {
      span.addEventListener("click", () => seekAndPlay(segments[+span.dataset.i].start));
    }
  } else {
    segments.forEach((seg, i) => {
      const row = document.createElement("div");
      row.className = `segment-row ${hitClass(i)}`;
      const time = document.createElement("span");
      time.className = "segment-time";
      if (viewFormat === "srt") {
        time.textContent = `${i + 1}\n${formatSrtTime(seg.start)} --> ${formatSrtTime(seg.end)}`;
        time.classList.add("multi-line");
      } else {
        time.textContent = `[${formatClock(seg.start)} → ${formatClock(seg.end)}]`;
      }
      const text = document.createElement("span");
      text.className = "segment-text";
      text.innerHTML = highlight(seg.text, q);
      row.append(time, text);
      row.addEventListener("click", () => seekAndPlay(seg.start));
      wrap.appendChild(row);
    });
  }
  renderSearchCount();
}

// ---------------------------------------------------------------- run
let pollTimer = null;
/** How many backend log lines have already been rendered. */
let logShown = 0;

function renderLogs(logs) {
  const box = $("log-box");
  for (let i = logShown; i < logs.length; i++) {
    const line = document.createElement("div");
    line.className = "log-line";
    line.textContent = `[${new Date().toLocaleTimeString()}] ${logs[i]}`;
    box.appendChild(line);
  }
  logShown = logs.length;
  box.scrollTop = box.scrollHeight;
}

function renderStats(st) {
  const dur = st.audio_duration_secs || 0;
  const elapsed = st.elapsed_secs || 0;
  const decode = st.decode_secs || 0;
  const rtf = dur > 0 ? elapsed / dur : 0;
  $("stats").textContent =
    `Audio duration: ${dur.toFixed(3)} s · Decode time: ${decode.toFixed(3)} s · ` +
    `Elapsed: ${elapsed.toFixed(3)} s · RTF = ${rtf.toFixed(3)}`;
}

async function poll() {
  try {
    const st = await invoke("get_progress");
    $("status").textContent = st.status;
    $("progress-bar").style.width = `${st.percent}%`;
    segments = st.segments.map((s) => ({
      index: s.index,
      start: s.start,
      end: s.end,
      text: s.text,
    }));
    renderLogs(st.logs || []);
    renderStats(st);
    recomputeMatches();
    renderResult();
    updateExportButtons();

    const finished =
      st.percent >= 100 || /^(Error|Stopped|Cancelled|No speech)/.test(st.status || "");
    if (finished) {
      clearInterval(pollTimer);
      pollTimer = null;
      $("start").disabled = false;
      $("cancel").disabled = true;
    }
  } catch (e) {
    clearInterval(pollTimer);
    pollTimer = null;
    showError(String(e));
    appendLog(`Error: ${e}`);
    $("start").disabled = false;
    $("cancel").disabled = true;
    updateExportButtons();
  }
}

$("start").addEventListener("click", async () => {
  showError("");
  // Warn about a missing file first — nothing can run without an input.
  if (!selectedPath) {
    const msg =
      'Warning: no file selected. Please click "Select audio or video file…" and choose a file first.';
    showError(msg);
    $("status").textContent = msg;
    appendLog(msg);
    return;
  }
  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) {
    // Users must provide an API key before anything can run.
    showError("Please provide a Doubao API key first.");
    apiKeyInput.focus();
    return;
  }
  $("start").disabled = true;
  $("cancel").disabled = false;
  $("status").textContent = "Starting...";
  // Optional range: empty/0 fields mean "the whole file".
  const startSec = parseFloat($("range-start").value) || 0;
  const endSec = parseFloat($("range-end").value) || 0;
  try {
    await invoke("recognize_file", {
      path: selectedPath,
      apiKey,
      startSec,
      endSec,
    });
    // Only clear the display once a new run has really started. If the run
    // was refused (e.g. the .srt/.txt already exist), whatever is shown —
    // e.g. results loaded from those files — stays untouched.
    resetResult();
    pollTimer = setInterval(poll, 500);
  } catch (e) {
    showError(String(e));
    $("status").textContent = String(e);
    appendLog(String(e));
    $("start").disabled = false;
    $("cancel").disabled = true;
    updateExportButtons();
  }
});

// Stop: keep everything recognized so far (segments stay on screen and can
// still be exported).
$("cancel").addEventListener("click", async () => {
  await invoke("cancel_recognition");
  $("status").textContent = "Stopping — keeping results so far...";
});

// ---------------------------------------------------------------- export
async function exportAs(format, defaultName) {
  if (!selectedPath || segments.length === 0) return;
  const path = await save({
    defaultPath: defaultName,
    filters: [{ name: "Text", extensions: [defaultName.split(".").pop()] }],
  });
  if (!path) return;
  try {
    await invoke("export_text", { path, format });
    $("status").textContent = `Saved to ${path}`;
  } catch (e) {
    showError(String(e));
  }
}

function baseName(path) {
  const name = path.split(/[\\/]/).pop();
  const idx = name.lastIndexOf(".");
  return idx > 0 ? name.slice(0, idx) : name;
}

$("export-srt").addEventListener("click", () =>
  exportAs("srt", `${baseName(selectedPath)}.srt`),
);
$("export-timestamped").addEventListener("click", () =>
  exportAs("timestamped_txt", `${baseName(selectedPath)}.txt`),
);
$("export-plain").addEventListener("click", () =>
  exportAs("plain_txt", `${baseName(selectedPath)}.txt`),
);

// ---------------------------------------------------------------- share
// Android/iOS only (native share sheet, e.g. WeChat); hidden on desktop.
async function shareAs(format) {
  if (segments.length === 0) return;
  try {
    const name = selectedPath ? baseName(selectedPath) : "transcript";
    await invoke("share_result", { format, name });
  } catch (e) {
    showError(String(e));
  }
}

for (const [id, fmt] of [
  ["share-srt", "srt"],
  ["share-timestamped", "timestamped_txt"],
  ["share-plain", "plain_txt"],
]) {
  const el = $(id);
  if (el) el.addEventListener("click", () => shareAs(fmt));
}

// ---------------------------------------------------------------- info
async function loadInfo() {
  try {
    const info = await invoke("get_info");
    $("info-app-version").textContent = info.app_version;
    $("info-version").textContent = info.sherpa_onnx_version;
    $("info-git-sha1").textContent = info.git_sha1;
    $("info-git-date").textContent = info.git_date;
    $("info-onnxruntime").textContent = info.onnxruntime_version;
    $("info-model-file").textContent = info.model_file;
    wireLink("info-model-url", info.model_url);
    wireLink("info-model-doc", info.model_doc_url);
    wireLink("info-api-doc", info.api_doc_url);
    wireLink("api-doc-link", info.api_doc_url);
    wireLink("pricing-link", info.pricing_url);
    // Sharing goes through the system share sheet — mobile only.
    $("share-row").classList.toggle("hidden", !info.is_mobile);
  } catch (e) {
    console.error("get_info failed", e);
  }
}

wireLink("info-github", "https://github.com/k2-fsa/sherpa-onnx");
wireLink("info-dev", "https://github.com/csukuangfj");
wireLink("info-docs", "https://k2-fsa.github.io/sherpa/onnx/");
wireLink("info-releases", "https://github.com/k2-fsa/sherpa-onnx/releases");
wireLink("info-wechat", "https://k2-fsa.github.io/sherpa/social-groups.html");
wireLink("info-footer-link", "https://github.com/k2-fsa/sherpa-onnx");
loadInfo();
