const { invoke } = window.__TAURI__.core;
const { open } = window.__TAURI__.dialog;

const selectBtn = document.querySelector("#select-btn");
const statusEl = document.querySelector("#status");
const resultsEl = document.querySelector("#results");
const resultsBody = document.querySelector("#results-body");

selectBtn.addEventListener("click", async () => {
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

  if (selected === null) {
    return;
  }

  statusEl.textContent = "Recognizing... please wait";
  statusEl.className = "status status-working";
  selectBtn.disabled = true;
  resultsEl.style.display = "none";

  try {
    const segments = await invoke("recognize_file", { path: selected });

    if (segments.length === 0) {
      statusEl.textContent = "No speech detected.";
      statusEl.className = "status";
      return;
    }

    statusEl.textContent = `Done. Found ${segments.length} segment(s).`;
    statusEl.className = "status status-done";

    resultsBody.innerHTML = "";
    for (const seg of segments) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${seg.start.toFixed(2)}s</td>
        <td>${seg.end.toFixed(2)}s</td>
        <td>${escapeHtml(seg.text)}</td>
      `;
      resultsBody.appendChild(tr);
    }
    resultsEl.style.display = "block";
  } catch (err) {
    statusEl.textContent = `Error: ${err}`;
    statusEl.className = "status status-error";
  } finally {
    selectBtn.disabled = false;
  }
});

function escapeHtml(text) {
  const el = document.createElement("span");
  el.textContent = text;
  return el.innerHTML;
}
