const { invoke, convertFileSrc } = window.__TAURI__.core;
const { save, ask } = window.__TAURI__.dialog;
const { open: openUrl } = window.__TAURI__.shell;

const startBtn = document.querySelector("#start-btn");
const stopBtn = document.querySelector("#stop-btn");
const clearBtn = document.querySelector("#clear-btn");
const statusEl = document.querySelector("#status");
const recordingIndicator = document.querySelector("#recording-indicator");
const elapsedTimer = document.querySelector("#elapsed-timer");
const resultsEl = document.querySelector("#results");
const resultsBody = document.querySelector("#results-body");
const copyTextBtn = document.querySelector("#copy-text-btn");
const copyTimedBtn = document.querySelector("#copy-timed-btn");
const exportSrtBtn = document.querySelector("#export-srt-btn");
const saveAllBtn = document.querySelector("#save-all-btn");
const playerWrapper = document.querySelector("#player-wrapper");
const player = document.querySelector("#player");
const settingsBtn = document.querySelector("#settings-btn");
const settingsModal = document.querySelector("#settings-modal");
const setThreshold = document.querySelector("#set-threshold");
const setMinSilence = document.querySelector("#set-min-silence");
const setMinSpeech = document.querySelector("#set-min-speech");
const setMaxSpeech = document.querySelector("#set-max-speech");
const setNumThreads = document.querySelector("#set-num-threads");
const settingsApplyBtn = document.querySelector("#settings-apply");
const settingsCancelBtn = document.querySelector("#settings-cancel");
const deviceSelect = document.querySelector("#device-select");

let recording = false;
let pollTimer = null;
let elapsedInterval = null;
let lastSegments = [];
let modelsReady = false;
let recordingStartTime = null;
let hasDevices = false;

// ---------------------------------------------------------------------------
// Model initialization polling
// ---------------------------------------------------------------------------

startBtn.disabled = true;
statusEl.textContent = "Loading models...";
statusEl.className = "status status-working";

async function loadDevices() {
  try {
    const devices = await invoke("list_input_devices");
    deviceSelect.innerHTML = "";
    if (devices.length === 0) {
      hasDevices = false;
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "No microphone found";
      deviceSelect.appendChild(opt);
      deviceSelect.disabled = true;
      startBtn.disabled = true;
      statusEl.textContent = "No microphone detected. Please connect a microphone and restart.";
      statusEl.className = "status status-error";
      return;
    }
    hasDevices = true;
    const selected = await invoke("get_selected_device");
    devices.forEach((d) => {
      const opt = document.createElement("option");
      opt.value = d.name;
      opt.textContent = d.is_default ? `${d.name} (default)` : d.name;
      if (selected ? d.name === selected : d.is_default) {
        opt.selected = true;
      }
      deviceSelect.appendChild(opt);
    });
    deviceSelect.disabled = false;
    if (modelsReady) {
      startBtn.disabled = false;
      statusEl.textContent = "";
      statusEl.className = "status";
    }
  } catch (err) {
    deviceSelect.innerHTML = "<option>Error loading devices</option>";
    deviceSelect.disabled = true;
  }
}

deviceSelect.addEventListener("change", async () => {
  const name = deviceSelect.value || null;
  try {
    await invoke("set_input_device", { deviceName: name });
  } catch (err) {
    flashStatus(`Device error: ${err}`, true);
  }
});

loadDevices();

function pollInitStatus() {
  invoke("get_init_status")
    .then((res) => {
      if (res.status === 1) {
        modelsReady = true;
        startBtn.disabled = !hasDevices;
        settingsBtn.disabled = false;
        clearBtn.disabled = false;
        if (hasDevices) {
          statusEl.textContent = "";
          statusEl.className = "status";
        }
      } else if (res.status === 2) {
        startBtn.disabled = true;
        settingsBtn.disabled = true;
        statusEl.textContent = `Initialization failed: ${res.error}`;
        statusEl.className = "status status-error";
      } else {
        setTimeout(pollInitStatus, 300);
      }
    })
    .catch((err) => {
      startBtn.disabled = true;
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
  const text = lastSegments.map((s) => s.text).join("\n");
  await navigator.clipboard.writeText(text);
  flashStatus("Text copied.");
});

copyTimedBtn.addEventListener("click", async () => {
  const lines = lastSegments.map(
    (s) => `[${s.wall_start} --> ${s.wall_end}] ${s.text}`
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

saveAllBtn.addEventListener("click", async () => {
  const filePath = await save({
    defaultPath: "recording.wav",
    filters: [{ name: "WAV Audio", extensions: ["wav"] }],
  });

  if (filePath === null) return;

  try {
    await invoke("save_all_audio", { path: filePath });
    flashStatus(`Audio saved to: ${filePath}`, false, 8000);
  } catch (err) {
    flashStatus(`Save error: ${err}`, true);
  }
});

// ---------------------------------------------------------------------------
// Audio player & row sync
// ---------------------------------------------------------------------------

let lastActiveIdx = -1;

player.addEventListener("timeupdate", () => {
  const t = player.currentTime;
  const activeIdx = findSegmentIndex(t);

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
});

player.addEventListener("ended", () => {
  lastActiveIdx = -1;
  resultsBody
    .querySelectorAll("tr.active")
    .forEach((tr) => tr.classList.remove("active"));
});

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

// Click a table row to seek to that segment
resultsBody.addEventListener("click", (e) => {
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
  const wallPart = seg.wall_start.replace(/[:\s]/g, "-");
  const textPart = seg.text
    .replace(/[^\w一-鿿]/g, "_")
    .slice(0, 30);
  const defaultName = `segment-${idx + 1}-${wallPart}-${textPart}.wav`;

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
// Start / Stop recording
// ---------------------------------------------------------------------------

startBtn.addEventListener("click", async () => {
  if (recording || !modelsReady) return;

  try {
    await invoke("start_recording");
  } catch (err) {
    flashStatus(`Start error: ${err}`, true);
    return;
  }

  recording = true;
  startBtn.style.display = "none";
  stopBtn.style.display = "";
  settingsBtn.disabled = true;
  clearBtn.disabled = true;
  deviceSelect.disabled = true;
  recordingIndicator.style.display = "flex";
  playerWrapper.style.display = "none";
  player.src = "";
  lastActiveIdx = -1;
  statusEl.textContent = "";
  statusEl.className = "status";

  recordingStartTime = Date.now();
  elapsedInterval = setInterval(updateElapsedTimer, 1000);
  startPolling();
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  await invoke("stop_recording");
});

clearBtn.addEventListener("click", async () => {
  if (recording) return;

  const confirmed = await ask(
    "This will clear all recognition results and recorded audio. Continue?",
    { title: "Clear All", kind: "warning" }
  );
  if (!confirmed) return;

  try {
    await invoke("clear_results");
    lastSegments = [];
    resultsBody.innerHTML = "";
    resultsEl.style.display = "none";
    playerWrapper.style.display = "none";
    player.src = "";
    lastActiveIdx = -1;
    statusEl.textContent = "";
    statusEl.className = "status";
    flashStatus("Cleared.");
  } catch (err) {
    flashStatus(`Clear error: ${err}`, true);
  }
});

function updateElapsedTimer() {
  if (!recordingStartTime) return;
  const secs = Math.floor((Date.now() - recordingStartTime) / 1000);
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  elapsedTimer.textContent =
    String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const state = await invoke("get_recording_state");

      // Append new rows
      const prevLen = lastSegments.length;
      lastSegments = state.segments;
      for (let i = prevLen; i < state.segments.length; i++) {
        const seg = state.segments[i];
        const duration = (seg.end - seg.start).toFixed(2);
        const tr = document.createElement("tr");
        tr.dataset.idx = i;
        tr.innerHTML = `
          <td>${escapeHtml(stripYear(seg.wall_start))}</td>
          <td>${escapeHtml(stripYear(seg.wall_end))}</td>
          <td>${duration}s</td>
          <td>${escapeHtml(seg.text)}</td>
          <td><button class="save-seg-btn" data-idx="${i}" title="Save as WAV">&#128190;</button></td>
        `;
        resultsBody.appendChild(tr);
      }
      if (state.segments.length > prevLen) {
        resultsEl.style.display = "block";
        const lastRow = resultsBody.lastElementChild;
        if (lastRow) lastRow.scrollIntoView({ behavior: "smooth", block: "end" });
      }

      // Check if recording stopped
      if (!state.recording && recording) {
        clearInterval(pollTimer);
        pollTimer = null;
        if (elapsedInterval) {
          clearInterval(elapsedInterval);
          elapsedInterval = null;
        }
        recording = false;
        startBtn.style.display = "";
        stopBtn.style.display = "none";
        stopBtn.disabled = false;
        settingsBtn.disabled = false;
        clearBtn.disabled = false;
        deviceSelect.disabled = false;
        recordingIndicator.style.display = "none";

        const totalSecs = state.elapsed_secs;
        statusEl.textContent = `Done. ${state.segments.length} segment(s) in ${totalSecs.toFixed(1)}s.`;
        statusEl.className = "status status-done";

        // Load recorded audio for playback
        try {
          const audioPath = await invoke("get_recorded_audio_path");
          player.src = convertFileSrc(audioPath);
          playerWrapper.style.display = "block";
        } catch (err) {
          if (state.segments.length > 0) {
            flashStatus(`Could not load playback: ${err}`, true);
          }
        }
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      if (elapsedInterval) {
        clearInterval(elapsedInterval);
        elapsedInterval = null;
      }
      recording = false;
      startBtn.style.display = "";
      stopBtn.style.display = "none";
      stopBtn.disabled = false;
      settingsBtn.disabled = false;
      clearBtn.disabled = false;
      deviceSelect.disabled = false;
      recordingIndicator.style.display = "none";
      statusEl.textContent = `Poll error: ${err}`;
      statusEl.className = "status status-error";
    }
  }, 200);
}

// ---------------------------------------------------------------------------
// Settings modal
// ---------------------------------------------------------------------------

settingsBtn.addEventListener("click", async () => {
  if (!modelsReady || recording) return;

  try {
    const s = await invoke("get_settings");
    setThreshold.value = s.threshold;
    setMinSilence.value = s.min_silence_duration;
    setMinSpeech.value = s.min_speech_duration;
    setMaxSpeech.value = s.max_speech_duration;
    setNumThreads.value = s.num_threads;
    settingsModal.style.display = "flex";
  } catch (err) {
    flashStatus(`Settings error: ${err}`, true);
  }
});

settingsCancelBtn.addEventListener("click", () => {
  settingsModal.style.display = "none";
});

settingsModal.addEventListener("click", (e) => {
  if (e.target === settingsModal) {
    settingsModal.style.display = "none";
  }
});

settingsApplyBtn.addEventListener("click", async () => {
  const newSettings = {
    threshold: parseFloat(setThreshold.value),
    min_silence_duration: parseFloat(setMinSilence.value),
    min_speech_duration: parseFloat(setMinSpeech.value),
    max_speech_duration: parseFloat(setMaxSpeech.value),
    num_threads: parseInt(setNumThreads.value, 10),
  };

  if (
    isNaN(newSettings.threshold) ||
    newSettings.threshold < 0 ||
    newSettings.threshold > 1
  ) {
    flashStatus("Threshold must be between 0.0 and 1.0", true);
    return;
  }
  if (isNaN(newSettings.min_silence_duration) || newSettings.min_silence_duration < 0) {
    flashStatus("Min silence duration must be >= 0", true);
    return;
  }
  if (isNaN(newSettings.min_speech_duration) || newSettings.min_speech_duration < 0) {
    flashStatus("Min speech duration must be >= 0", true);
    return;
  }
  if (isNaN(newSettings.max_speech_duration) || newSettings.max_speech_duration <= 0) {
    flashStatus("Max speech duration must be > 0", true);
    return;
  }
  if (
    isNaN(newSettings.num_threads) ||
    newSettings.num_threads < 1 ||
    newSettings.num_threads > 16
  ) {
    flashStatus("Threads must be between 1 and 16", true);
    return;
  }

  settingsApplyBtn.disabled = true;
  try {
    await invoke("apply_settings", { newSettings });
    settingsModal.style.display = "none";

    modelsReady = false;
    startBtn.disabled = true;
    settingsBtn.disabled = true;
    statusEl.textContent = "Reloading models...";
    statusEl.className = "status status-working";
    pollInitStatus();
  } catch (err) {
    flashStatus(`Apply error: ${err}`, true);
  } finally {
    settingsApplyBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function escapeHtml(text) {
  const el = document.createElement("span");
  el.textContent = text;
  return el.innerHTML;
}

// "2026-04-21 18:52:28" -> "18:52:28" (time only for display)
function stripYear(wall) {
  const parts = wall.split(" ");
  return parts.length >= 2 ? parts[parts.length - 1] : wall;
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
