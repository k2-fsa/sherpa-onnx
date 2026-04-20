const { invoke, convertFileSrc } = window.__TAURI__.core;
const { open, save } = window.__TAURI__.dialog;
const { open: openUrl } = window.__TAURI__.shell;

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
const playerWrapper = document.querySelector("#player-wrapper");
const player = document.querySelector("#player");
const subtitleOverlay = document.querySelector("#subtitle-overlay");
const statsEl = document.querySelector("#stats");

let recognizing = false;
let pollTimer = null;
let lastSegments = [];
let modelsReady = false;
let modelThreads = 0;

// ---------------------------------------------------------------------------
// Model initialization polling
// ---------------------------------------------------------------------------

selectBtn.disabled = true;
statusEl.textContent = "Loading models...";
statusEl.className = "status status-working";

function pollInitStatus() {
  invoke("get_init_status")
    .then((res) => {
      if (res.status === 1) {
        // ready
        modelsReady = true;
        modelThreads = res.num_threads;
        selectBtn.disabled = false;
        statusEl.textContent = "";
        statusEl.className = "status";
      } else if (res.status === 2) {
        // error
        selectBtn.disabled = true;
        statusEl.textContent = `Initialization failed: ${res.error}`;
        statusEl.className = "status status-error";
      } else {
        // still pending, poll again
        setTimeout(pollInitStatus, 300);
      }
    })
    .catch((err) => {
      selectBtn.disabled = true;
      statusEl.textContent = `Init poll error: ${err}`;
      statusEl.className = "status status-error";
    });
}

pollInitStatus();

// ---------------------------------------------------------------------------
// External links
// ---------------------------------------------------------------------------

document.querySelectorAll("a[href]").forEach((a) => {
  a.addEventListener("click", (e) => {
    e.preventDefault();
    openUrl(e.currentTarget.href);
  });
});

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
// Player / subtitle / row sync
// ---------------------------------------------------------------------------

function setupPlayer(filePath) {
  const url = convertFileSrc(filePath);
  player.src = url;
  playerWrapper.style.display = "block";
  subtitleOverlay.textContent = "";
}

// Find segment index for a given time using binary search.
// Segments are sorted by start time.
function findSegmentIndex(t) {
  let lo = 0;
  let hi = lastSegments.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const seg = lastSegments[mid];
    if (t < seg.start) {
      hi = mid - 1;
    } else if (t >= seg.end) {
      lo = mid + 1;
    } else {
      return mid;
    }
  }
  return -1;
}

// Use rAF for smooth subtitle/row sync (~60fps instead of timeupdate's ~4fps).
let rafId = null;
let lastActiveIdx = -1;

function tick() {
  if (player.paused || player.ended) {
    rafId = null;
    return;
  }

  const t = player.currentTime;
  const activeIdx = findSegmentIndex(t);

  // Update subtitle
  subtitleOverlay.textContent =
    activeIdx >= 0 ? lastSegments[activeIdx].text : "";

  // Update row highlight (only on change to avoid constant reflows)
  if (activeIdx !== lastActiveIdx) {
    const rows = resultsBody.querySelectorAll("tr");
    if (lastActiveIdx >= 0 && lastActiveIdx < rows.length) {
      rows[lastActiveIdx].classList.remove("active");
    }
    if (activeIdx >= 0 && activeIdx < rows.length) {
      rows[activeIdx].classList.add("active");
      rows[activeIdx].scrollIntoView({ block: "nearest" });
    }
    lastActiveIdx = activeIdx;
  }

  rafId = requestAnimationFrame(tick);
}

function startSync() {
  if (rafId) return;
  lastActiveIdx = -1;
  rafId = requestAnimationFrame(tick);
}

function stopSync() {
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

player.addEventListener("play", startSync);
player.addEventListener("pause", () => {
  stopSync();
  // Still update subtitle on pause so it shows the correct text
  const t = player.currentTime;
  const idx = findSegmentIndex(t);
  subtitleOverlay.textContent = idx >= 0 ? lastSegments[idx].text : "";
});

player.addEventListener("ended", () => {
  stopSync();
  subtitleOverlay.textContent = "";
  lastActiveIdx = -1;
  resultsBody
    .querySelectorAll("tr.active")
    .forEach((tr) => tr.classList.remove("active"));
});

// Click a table row to seek to that segment
resultsBody.addEventListener("click", (e) => {
  // Handle save button click
  if (e.target.closest(".save-seg-btn")) {
    e.stopPropagation();
    const btn = e.target.closest(".save-seg-btn");
    const idx = parseInt(btn.dataset.idx, 10);
    if (idx >= 0 && idx < lastSegments.length) {
      saveSegment(idx);
    }
    return;
  }

  const tr = e.target.closest("tr");
  if (!tr) return;
  const idx = parseInt(tr.dataset.idx, 10);
  if (idx >= 0 && idx < lastSegments.length) {
    player.pause();
    // Seek 0.3s before the segment start to avoid missing the beginning
    const t = Math.max(0, lastSegments[idx].start - 0.3);
    player.currentTime = t;
    player.addEventListener(
      "seeked",
      () => {
        player.play().catch(() => {});
      },
      { once: true }
    );
  }
});

async function saveSegment(idx) {
  const seg = lastSegments[idx];
  const start = seg.start.toFixed(2).replace(".", "_");
  const end = seg.end.toFixed(2).replace(".", "_");
  const textPart = seg.text
    .replace(/[^\w\u4e00-\u9fff]/g, "_")
    .slice(0, 30);
  const defaultName = `segment-${idx + 1}-${start}s-${end}s-${textPart}.wav`;

  const filePath = await save({
    defaultPath: defaultName,
    filters: [{ name: "WAV Audio", extensions: ["wav"] }],
  });

  if (filePath === null) return;

  try {
    await invoke("save_segment_as_wav", {
      path: filePath,
      start: seg.start,
      end: seg.end,
    });
    flashStatus(`Saved: ${filePath}`, false, 8000);
  } catch (err) {
    flashStatus(`Save error: ${err}`, true);
  }
}

// ---------------------------------------------------------------------------
// File selection & recognition
// ---------------------------------------------------------------------------

selectBtn.addEventListener("click", async () => {
  if (recognizing || !modelsReady) return;

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
  statsEl.style.display = "none";
  playerWrapper.style.display = "none";
  player.src = "";
  lastSegments = [];
  statusEl.textContent = "Decoding audio file...";
  statusEl.className = "status status-working";

  try {
    await invoke("recognize_file", { path: selected });
    setupPlayer(selected);
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
      for (let i = 0; i < state.segments.length; i++) {
        const seg = state.segments[i];
        const tr = document.createElement("tr");
        tr.dataset.idx = i;
        tr.innerHTML = `
          <td>${seg.start.toFixed(2)}s</td>
          <td>${seg.end.toFixed(2)}s</td>
          <td>${escapeHtml(seg.text)}</td>
          <td><button class="save-seg-btn" data-idx="${i}" title="Save as WAV">&#128190;</button></td>
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
