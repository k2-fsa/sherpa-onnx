// Copyright (c)  2026  Xiaomi Corporation
//
// Plain-web demo for forced alignment over sherpa-onnx-wasm-main-fa.
// The model directories in assets/ are mounted at '/' in the wasm FS; pick
// one by its index (sorted directory names).
'use strict'

let Module = null;
let aligner = null;
let audioCtx = null;
let decoded = null;      // {samples: Float32Array, sampleRate: 16000, duration}
let lastResult = null;

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

function log(msg) {
  console.log('[fa]', msg);
  setStatus(msg);
}

// The model index into the directories of assets/ (developer constant, like
// wasm/tts's modelType; set at build/CI time — not exposed to users).
const modelIndex = 0;

// ---- model selection by index ------------------------------------------

function listModelDirs() {
  // The preloaded assets are mounted at '/'; model dirs are the directories
  // that contain meta.json.
  const candidates = ['zh-aishell3-pinyin', 'zh-biaobei-pinyin', 'zh-large-pinyin',
                      'english-mfa-v2.2.1', 'english-mfa-v3.0.0', 'english-mfa-v3.1.0'];
  return candidates.filter((d) => {
    try {
      return Module.FS.readdir('/' + d).indexOf('meta.json') >= 0;
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
        {modelDir: dir, lexicon: dir + '/lexicon.txt'}, Module);
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
}

// ---- alignment ----------------------------------------------------------

function align() {
  if (!aligner) {
    setStatus('Please load the model first');
    return;
  }
  if (!decoded) {
    setStatus('Please select an audio file first');
    return;
  }
  const text = document.getElementById('text').value.trim();
  if (!text) {
    setStatus('Please enter the transcript');
    return;
  }
  const t0 = performance.now();
  const stream = aligner.createStream();
  stream.acceptWaveform(decoded.sampleRate, decoded.samples);
  lastResult = aligner.compute(stream, text);
  stream.free();
  const elapsed = (performance.now() - t0) / 1000;

  const words = [];
  for (const tier of lastResult.tiers) {
    if (tier.name === 'words') {
      for (const itv of tier.intervals) {
        if (itv.text) words.push(itv);
      }
    }
  }
  if (words.length === 0) {
    setStatus('Alignment failed: check that the transcript matches the audio');
    return;
  }
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

function downloadTextGrid() {
  if (!lastResult) return;
  const blob = new Blob([lastResult.textgrid], {type: 'text/plain'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'result.TextGrid';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- wiring -------------------------------------------------------------

window.onSherpaReady = () => {
  Module = window.Module;
  log('wasm engine ready');
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
      bytes = Module.FS.readFile('/' + dir + '/' + wav);
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
  document.getElementById('download-textgrid').onclick = downloadTextGrid;
};
