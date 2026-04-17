const { invoke } = window.__TAURI__.core;
const { open, save } = window.__TAURI__.dialog;

const selectBtn = document.querySelector("#select-btn");
const cancelBtn = document.querySelector("#cancel-btn");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const resultsBody = document.querySelector("#results-body");
const progressContainer = document.querySelector("#progress-container");
const progressBar = document.querySelector("#progress-bar");
const progressLabel = document.querySelector("#progress-label");
const copyTextBtn = document.querySelector("#copy-text-btn");
const copyTimedBtn = document.querySelector("#copy-timed-btn");
const exportSrtBtn = document.querySelector("#export-srt-btn");

let recognizing = false;
let pollTimer = null;
let lastSegments = [];

// ---------------------------------------------------------------------------
// Copy / Export handlers
// ---------------------------------------------------------------------------

copyTextBtn.addEventListener("click", async () => {
  const text = lastSegments.map((s) => s.text).join("");
  await navigator.clipboard.writeText(text);
  flashStatus("Text copied.");
});

copyTimedBtn.addEventListener("click", async () => {
  const lines = lastSegments.map(
    (s) => `[${formatTime(s.start)} --> ${formatTime(s.end)}] ${s.text}`
  );
  await navigator.clipboard.writeText(lines.join("\n"));
  flashStatus("Text with time copied.");
});

exportSrtBtn.addEventListener("click", async () => {
  const filePath = await save({
    defaultPath: "subtitles.srt",
    filters: [{ name: "SubRip", extensions: ["srt"] }],
  });

  if (filePath === null) return;

  try {
    await invoke("export_srt", { path: filePath });
    flashStatus(`SRT saved to: ${filePath}`, false, 8000);
  } catch (err) {
    flashStatus(`Export error: ${err}`, true);
  }
});

// ---------------------------------------------------------------------------
// File selection & recognition
// ---------------------------------------------------------------------------

selectBtn.addEventListener("click", async () => {
  if (recognizing) return;

  const selected = await open({
    multiple: false,
    filters: [
      {
        name: "Audio",
        extensions: ["wav", "mp3", "flac", "ogg", "aac", "m4a", "aiff", "caf"],
      },
      {
        name: "Video",
        extensions: ["mp4", "mkv", "webm", "avi", "mov"],
      },
    ],
  });

  if (selected === null) return;

  // Reset UI
  recognizing = true;
  selectBtn.disabled = true;
  cancelBtn.style.display = "";
  cancelBtn.disabled = false;
  progressContainer.style.display = "flex";
  progressBar.style.width = "0%";
  progressLabel.textContent = "0%";
  resultsEl.style.display = "none";
  resultsBody.innerHTML = "";
  lastSegments = [];
  statusEl.textContent = "Decoding audio file...";
  statusEl.className = "status status-working";

  try {
    await invoke("recognize_file", { path: selected });
    startPolling();
  } catch (err) {
    recognizing = false;
    selectBtn.disabled = false;
    cancelBtn.style.display = "none";
    progressContainer.style.display = "none";
    statusEl.textContent = `Error: ${err}`;
    statusEl.className = "status status-error";
  }
});

cancelBtn.addEventListener("click", async () => {
  await invoke("cancel_recognition");
  cancelBtn.disabled = true;
});

// ---------------------------------------------------------------------------
// Progress polling
// ---------------------------------------------------------------------------

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const state = await invoke("get_recognition_progress");

      // Update progress bar
      progressBar.style.width = state.percent + "%";
      progressLabel.textContent = state.percent + "%";

      // Update results table
      lastSegments = state.segments;
      resultsBody.innerHTML = "";
      for (const seg of state.segments) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${seg.start.toFixed(2)}s</td>
          <td>${seg.end.toFixed(2)}s</td>
          <td>${escapeHtml(seg.text)}</td>
        `;
        resultsBody.appendChild(tr);
      }
      if (state.segments.length > 0) {
        resultsEl.style.display = "block";
      }

      // Check terminal states
      if (state.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        recognizing = false;
        selectBtn.disabled = false;
        cancelBtn.style.display = "none";
        progressBar.style.width = "100%";
        progressLabel.textContent = "100%";
        statusEl.textContent = `Done. Found ${state.segments.length} segment(s).`;
        statusEl.className = "status status-done";
      } else if (state.status === "cancelled") {
        clearInterval(pollTimer);
        pollTimer = null;
        recognizing = false;
        selectBtn.disabled = false;
        cancelBtn.style.display = "none";
        progressContainer.style.display = "none";
        statusEl.textContent = "Cancelled.";
        statusEl.className = "status status-cancelled";
      } else if (state.status.startsWith("error:")) {
        clearInterval(pollTimer);
        pollTimer = null;
        recognizing = false;
        selectBtn.disabled = false;
        cancelBtn.style.display = "none";
        progressContainer.style.display = "none";
        statusEl.textContent = state.status;
        statusEl.className = "status status-error";
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      recognizing = false;
      selectBtn.disabled = false;
      cancelBtn.style.display = "none";
      progressContainer.style.display = "none";
      statusEl.textContent = `Poll error: ${err}`;
      statusEl.className = "status status-error";
    }
  }, 200);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return (
    String(h).padStart(2, "0") +
    ":" +
    String(m).padStart(2, "0") +
    ":" +
    String(s).padStart(2, "0") +
    "," +
    String(ms).padStart(3, "0")
  );
}

function escapeHtml(text) {
  const el = document.createElement("span");
  el.textContent = text;
  return el.innerHTML;
}

let flashTimer = null;
function flashStatus(msg, isError, durationMs) {
  const prev = { text: statusEl.textContent, cls: statusEl.className };
  statusEl.textContent = msg;
  statusEl.className = isError ? "status status-error" : "status status-done";
  if (flashTimer) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    statusEl.textContent = prev.text;
    statusEl.className = prev.cls;
    flashTimer = null;
  }, durationMs || 2000);
}
