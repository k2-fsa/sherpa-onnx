// Copyright (c)  2026  Xiaomi Corporation
//
// Plain-web demo for forced alignment over sherpa-onnx-wasm-main-fa.
// The model directories in assets/ are mounted at '/' in the wasm FS; pick
// one by its index (sorted directory names).
'use strict'

// The emscripten module is called sherpaModule (not Module): a top-level
// `Module` in a classic script would collide with the glue's global
// `var Module`.

let sherpaModule = null;
let aligner = null;
let audioCtx = null;
let decoded = null;      // {samples: Float32Array, sampleRate: 16000, duration}
let lastResult = null;
let lastWords = [];
let fullSource = null;      // whole-file playback source
let recorder = null;        // MediaRecorder while recording
let recordChunks = [];
let recordTimer = null;
let recordStartMs = 0;

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

function log(msg) {
  console.log('[fa]', msg);
  setStatus(msg);
  appendLog(msg);
}

// Appends a line to the on-page Log section (also used for the wasm
// engine's own output, see index.html).
function appendLog(msg) {
  const box = document.getElementById('log-box');
  if (!box) return;
  const line = document.createElement('div');
  line.className = 'log-line';
  line.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
}
window.appendLog = appendLog;

// The model index into the directories of assets/ (developer constant, like
// wasm/tts's modelType; set at build/CI time — not exposed to users).
const modelIndex = 0;

// Fills the Version card and the download link of this demo. The zip file
// name embeds the sherpa-onnx version, see .github/workflows/wasm-fa.yaml.
function showVersionInfo() {
  const g = (f) =>
      sherpaModule['_' + f] ? sherpaModule.UTF8ToString(sherpaModule['_' + f]()) : 'unknown';
  const version = g('SherpaOnnxGetVersionStr');
  const lines = [
    'sherpa-onnx: ' + version,
    'Git SHA1: ' + g('SherpaOnnxGetGitSha1'),
    'Git date: ' + g('SherpaOnnxGetGitDate'),
    'onnxruntime: ' + g('SherpaOnnxGetOnnxruntimeVersionStr'),
  ];
  document.getElementById('version-info').innerHTML = lines.join('<br/>');

  const dirs = listModelDirs();
  const model = dirs[Math.min(modelIndex, dirs.length - 1)] || 'model';
  const zip = 'sherpa-onnx-v' + version + '-wasm-fa-' + model + '.zip';
  const url = 'https://github.com/k2-fsa/sherpa-onnx/releases/download/v' +
      version + '/' + zip;
  const dir = zip.replace(/\.zip$/, '');
  const a = document.getElementById('download-app');
  a.href = url;
  a.textContent = 'Download this demo: ' + zip;
  document.getElementById('run-locally').textContent = [
    'wget ' + url,
    'unzip ' + zip,
    'cd ' + dir,
    'python3 -m http.server 8000',
  ].join('\n');

  const copyBtn = document.getElementById('copy-code');
  copyBtn.onclick = () => {
    navigator.clipboard.writeText(document.getElementById('run-locally').textContent)
        .then(() => {
          copyBtn.textContent = 'Copied!';
          setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
        })
        .catch(() => { copyBtn.textContent = 'Copy failed'; });
  };
}

// ---- model selection by index ------------------------------------------

function listModelDirs() {
  // The preloaded assets are mounted at '/'; model dirs are the directories
  // that contain meta.json.
  const candidates = ['zh-aishell3-pinyin', 'zh-biaobei-pinyin', 'zh-large-pinyin',
                      'english-mfa-v2.2.1', 'english-mfa-v3.0.0', 'english-mfa-v3.1.0'];
  return candidates.filter((d) => {
    try {
      return sherpaModule.FS.readdir('/' + d).indexOf('meta.json') >= 0;
    } catch (e) {
      return false;
    }
  });
}

function loadModel() {
  const dirs = listModelDirs();
  if (dirs.length === 0) {
    setStatus('No model in assets/; please read assets/README.md first');
    return;
  }
  const dir = dirs[Math.min(modelIndex, dirs.length - 1)];
  if (aligner) {
    aligner.free();
    aligner = null;
  }
  try {
    aligner = new ForcedAligner(
        {modelDir: dir, lexicon: dir + '/lexicon.txt'}, sherpaModule);
    setStatus('Model loaded: ' + dir + ' (ready to align)');
  } catch (e) {
    setStatus('Failed to load the model: ' + e);
  }
}

// ---- audio decoding -----------------------------------------------------

async function decodeArrayBuffer(arrayBuffer) {
  if (!audioCtx) audioCtx = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  // mix down to mono and resample to 16 kHz
  const frameCount = Math.max(1, Math.ceil(audioBuffer.duration * 16000));
  const offline = new OfflineAudioContext(1, frameCount, 16000);
  const src = offline.createBufferSource();
  src.buffer = audioBuffer;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();
  return {
    samples: rendered.getChannelData(0),
    sampleRate: 16000,
    duration: rendered.duration,
  };
}

async function loadFile(file) {
  log('Decoding: ' + file.name + ' …');
  decoded = await decodeArrayBuffer(await file.arrayBuffer());
  log('Decoded: ' + decoded.duration.toFixed(2) + ' s');
  resetResult();
}

// ---- alignment ----------------------------------------------------------

function warnUser(msg) {
  alert(msg);
  setStatus(msg);
  appendLog(msg);
}

function align() {
  if (!aligner) {
    setStatus('Please load the model first');
    return;
  }
  if (!decoded) {
    warnUser('Please select or record an audio file first');
    return;
  }
  const text = document.getElementById('text').value.trim();
  if (!text) {
    warnUser('Please enter the transcript of the audio');
    return;
  }
  const t0 = performance.now();
  const stream = aligner.createStream();
  stream.acceptWaveform(decoded.sampleRate, decoded.samples);
  const computed = aligner.compute(stream, text);
  stream.free();
  const elapsed = (performance.now() - t0) / 1000;

  const words = [];
  for (const tier of computed.tiers) {
    if (tier.name === 'words') {
      for (const itv of tier.intervals) {
        if (itv.text) words.push(itv);
      }
    }
  }
  if (words.length === 0) {
    resetResult();
    log('Alignment failed: no words. Please check that the transcript matches the audio; see the Log for details.');
    return;
  }
  lastResult = computed;
  lastWords = words;
  updateDownloadButtons();
  document.getElementById('stats').textContent =
      'Words: ' + words.length + ' | Audio duration: ' +
      decoded.duration.toFixed(3) + ' s | Elapsed: ' + elapsed.toFixed(3) +
      ' s | RTF = ' + (elapsed / decoded.duration).toFixed(3);

  const tbody = document.querySelector('#table tbody');
  tbody.innerHTML = '';
  for (const w of words) {
    const tr = document.createElement('tr');
    const td1 = document.createElement('td');
    td1.textContent = w.text;
    const td2 = document.createElement('td');
    td2.textContent = '[' + w.start.toFixed(2) + ' -> ' + w.end.toFixed(2) + ']';
    const td3 = document.createElement('td');
    const btn = document.createElement('button');
    btn.textContent = '▶';
    btn.onclick = (e) => {
      e.stopPropagation();
      playWord(w.start, w.end);
    };
    td3.appendChild(btn);
    tr.append(td1, td2, td3);
    tr.onclick = () => playWord(w.start, w.end);
    tbody.appendChild(tr);
  }
  setStatus('Alignment complete');
}

// Sample-accurate playback of [start, end] seconds of the decoded audio.
function playWord(start, end) {
  const s = Math.max(0, Math.floor(start * decoded.sampleRate));
  const e = Math.min(decoded.samples.length, Math.ceil(end * decoded.sampleRate));
  const slice = decoded.samples.slice(s, e);
  const buf = audioCtx.createBuffer(1, slice.length, decoded.sampleRate);
  buf.getChannelData(0).set(slice);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  src.start();
}

/** Formats seconds as mm:ss.cc */
function formatClock(t) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return String(m).padStart(2, '0') + ':' + s.toFixed(2).padStart(5, '0');
}

// The export buttons are only enabled when a result exists.
function updateDownloadButtons() {
  const has = lastResult !== null && lastWords.length > 0;
  document.getElementById('download-textgrid').disabled = !has;
  document.getElementById('download-timestamped').disabled = !has;
}

function resetResult() {
  lastResult = null;
  lastWords = [];
  document.querySelector('#table tbody').innerHTML = '';
  document.getElementById('stats').textContent = '';
  updateDownloadButtons();
}

function downloadText(content, name) {
  const blob = new Blob([content], {type: 'text/plain'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function downloadTextGrid() {
  if (!lastResult) return;
  downloadText(lastResult.textgrid, 'result.TextGrid');
}

function downloadTimestamped() {
  if (lastWords.length === 0) return;
  downloadText(
      lastWords.map((w) => `[${formatClock(w.start)} -> ${formatClock(w.end)}] ${w.text}`).join('\n'),
      'result_timestamped.txt');
}

function stopFullPlayback() {
  if (fullSource) {
    try { fullSource.stop(); } catch (e) {}
    fullSource = null;
  }
  document.getElementById('play').textContent = 'Play audio';
}

// Toggle whole-file playback of the loaded audio.
function playAudio() {
  if (fullSource) {
    stopFullPlayback();
    return;
  }
  if (!decoded) {
    setStatus('Please select an audio file first');
    return;
  }
  if (!audioCtx) audioCtx = new AudioContext();
  const buf = audioCtx.createBuffer(1, decoded.samples.length, decoded.sampleRate);
  buf.getChannelData(0).set(decoded.samples);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  src.onended = () => stopFullPlayback();
  src.start();
  fullSource = src;
  document.getElementById('play').textContent = 'Stop audio';
}

// Toggle microphone recording; the recording is loaded for alignment.
async function toggleRecord() {
  const btn = document.getElementById('record');
  if (recorder) {
    recorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({audio: true});
    const rec = new MediaRecorder(stream);
    recordChunks = [];
    rec.ondataavailable = (e) => {
      if (e.data.size > 0) recordChunks.push(e.data);
    };
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      if (recordTimer) {
        clearInterval(recordTimer);
        recordTimer = null;
      }
      recorder = null;
      btn.textContent = 'Record';
      const blob = new Blob(recordChunks, {type: rec.mimeType || 'audio/webm'});
      if (blob.size === 0) {
        log('Recording discarded');
        return;
      }
      log('Recording finished: ' + (blob.size / 1024).toFixed(0) + ' KB');
      await loadFile(new File([blob], 'recording.webm'));
    };
    recorder = rec;
    rec.start();
    recordStartMs = Date.now();
    recordTimer = setInterval(() => {
      btn.textContent =
          'Stop recording (' + Math.floor((Date.now() - recordStartMs) / 1000) + 's)';
    }, 500);
    btn.textContent = 'Stop recording (0s)';
    log('Recording started');
  } catch (e) {
    setStatus('Cannot access the microphone: ' + e);
  }
}

function clearAll() {
  stopFullPlayback();
  if (recorder) {
    recordChunks = [];
    recorder.stop();
  }
  resetResult();
  decoded = null;
  document.getElementById('file').value = '';
  document.getElementById('text').value = '';
  setStatus('Ready');
}

// ---- wiring -------------------------------------------------------------

window.onSherpaReady = () => {
  sherpaModule = window.Module;
  log('wasm engine ready');
  showVersionInfo();
  loadModel();

  document.getElementById('file').onchange = (e) => {
    if (e.target.files.length > 0) {
      loadFile(e.target.files[0]);
    }
  };
  document.getElementById('example').onclick = async () => {
    const dirs = listModelDirs();
    const dir = dirs[Math.min(modelIndex, dirs.length - 1)];
    const wav = dir.indexOf('english') === 0 ? 'en.wav' : 'zh.wav';
    log('Loading example: ' + dir + '/' + wav);
    let bytes;
    try {
      bytes = sherpaModule.FS.readFile('/' + dir + '/' + wav);
    } catch (e) {
      setStatus('Example audio not found: ' + dir + '/' + wav);
      return;
    }
    await loadFile(new File([bytes], wav));
    // The test transcripts of the packages' test wavs (data, not UI).
    document.getElementById('text').value =
        dir.indexOf('english') === 0 ?
        'ask not what your country can do for you, ask what you can do for your country.' :
        '不要问你的国家能为你做什么，而要问你能为你的国家做什么。';
    };
  document.getElementById('align').onclick = align;
  document.getElementById('clear').onclick = clearAll;
  document.getElementById('play').onclick = playAudio;
  document.getElementById('record').onclick = toggleRecord;
  document.getElementById('download-textgrid').onclick = downloadTextGrid;
  document.getElementById('download-timestamped').onclick = downloadTimestamped;

};

