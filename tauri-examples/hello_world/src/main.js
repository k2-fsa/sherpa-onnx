const { invoke } = window.__TAURI__.core;

async function loadVersion() {
  const version = await invoke("get_version");
  const gitSha1 = await invoke("get_git_sha1");
  const gitDate = await invoke("get_git_date");
  const ortVersion = await invoke("get_onnxruntime_version");

  document.getElementById("version").textContent = version;
  document.getElementById("git-sha1").textContent = gitSha1;
  document.getElementById("git-date").textContent = gitDate;
  document.getElementById("ort-version").textContent = ortVersion;
}

window.addEventListener("DOMContentLoaded", loadVersion);
