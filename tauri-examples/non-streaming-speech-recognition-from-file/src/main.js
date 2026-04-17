const { invoke } = window.__TAURI__.core;
const { open } = window.__TAURI__.dialog;

const selectBtn = document.querySelector("#select-btn");
const cancelBtn = document.querySelector("#cancel-btn");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const resultsBody = document.querySelector("#results-body");
const progressContainer = document.querySelector("#progress-container");
const progressBar = document.querySelector("#progress-bar");
const progressLabel = document.querySelector("#progress-label");

let recognizing = false;
let pollTimer = null;

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
  statusEl.textContent = "Decoding audio file...";
  statusEl.className = "status status-working";

  try {
    await invoke("recognize_file", { path: selected });
    // Start polling for progress
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

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const state = await invoke("get_recognition_progress");

      // Update progress bar
      progressBar.style.width = state.percent + "%";
      progressLabel.textContent = state.percent + "%";

      // Update results table
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

function escapeHtml(text) {
  const el = document.createElement("span");
  el.textContent = text;
  return el.innerHTML;
}
