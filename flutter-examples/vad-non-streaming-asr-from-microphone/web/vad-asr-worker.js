// VAD + ASR Web Worker for real-time microphone input.
// Accepts audio chunks via 'acceptWaveform' messages and runs ASR on detected segments.
//
// Messages: Main Thread → Worker
//
//   init — Initialize WASM module, create VAD and ASR instances.
//     {
//       type:          'init',
//       jsGlueSource:  String,
//       vadJsSource:   String,
//       asrJsSource:   String,
//       wasmBinary:    ArrayBuffer,
//       vadModelFiles: Object,
//       asrModelFiles: Object,
//       vadConfig:     Object,
//       asrConfig:     Object,
//     }
//
//   acceptWaveform — Feed audio samples to VAD.
//     { type: 'acceptWaveform', samples: Float32Array }
//
//   reset — Reset VAD state for a new recording session.
//     { type: 'reset' }
//
//   dispose — Destroy instances and close worker.
//     { type: 'dispose' }
//
// Messages: Worker → Main Thread
//
//   ready — Initialization complete.
//     { type: 'ready' }
//
//   speechStateChanged — VAD speech detection state changed.
//     { type: 'speechStateChanged', isSpeaking: Boolean }
//
//   segmentDetected — A segment was detected and ASR finished.
//     { type: 'segmentDetected', index, start, end, samples, text, elapsedSeconds }
//
//   error — Error message.
//     { type: 'error', message: String }

let Module = null;
let vad = null;
let recognizer = null;
let segmentIndex = 0;

function getFS() {
  if (Module && Module.FS) return Module.FS;
  if (typeof FS !== 'undefined') return FS;
  throw new Error('FS not found');
}

function mkdirTree(path) {
  const fs = getFS();
  const parts = path.split('/');
  let current = '';
  for (const part of parts) {
    if (!part) continue;
    current = current + '/' + part;
    try { fs.mkdir(current); } catch (_) {}
  }
}

function writeFile(path, data) {
  getFS().writeFile(path, data);
}

function extractSegment(vadHandle, sampleRate) {
  const h = Module._SherpaOnnxVoiceActivityDetectorFront(vadHandle);
  const segStart = Module.HEAP32[h / 4];
  const samplesPtr = Module.HEAP32[h / 4 + 1] / 4;
  const numSegSamples = Module.HEAP32[h / 4 + 2];

  const segSamples = new Float32Array(numSegSamples);
  for (let j = 0; j < numSegSamples; j++) {
    segSamples[j] = Module.HEAPF32[samplesPtr + j];
  }

  Module._SherpaOnnxDestroySpeechSegment(h);

  return {
    start: segStart / sampleRate,
    end: segStart / sampleRate + numSegSamples / sampleRate,
    samples: segSamples,
  };
}

function processDetectedSegments() {
  const sampleRate = 16000;
  while (Module._SherpaOnnxVoiceActivityDetectorEmpty(vad.handle) !== 1) {
    const seg = extractSegment(vad.handle, sampleRate);
    const idx = segmentIndex++;

    // Run ASR immediately.
    const startTime = performance.now();
    const stream = recognizer.createStream();
    stream.acceptWaveform(sampleRate, seg.samples);
    recognizer.decode(stream);
    const result = recognizer.getResult(stream);
    stream.free();
    const elapsed = (performance.now() - startTime) / 1000;

    self.postMessage({
      type: 'segmentDetected',
      index: idx,
      start: seg.start,
      end: seg.end,
      samples: seg.samples,
      text: result.text || '',
      elapsedSeconds: elapsed,
    });

    Module._SherpaOnnxVoiceActivityDetectorPop(vad.handle);
  }
}

self.onmessage = async function(e) {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      // 1. Load Emscripten JS glue.
      if (msg.jsGlueSource) self.eval(msg.jsGlueSource);

      // 2. Module stub.
      self.eval('if (typeof module === "undefined") { var module = {}; }');

      // 3. Load VAD and ASR JS helpers.
      if (msg.vadJsSource) {
        const vadBlob = new Blob([msg.vadJsSource], { type: 'application/javascript' });
        const vadUrl = URL.createObjectURL(vadBlob);
        importScripts(vadUrl);
        URL.revokeObjectURL(vadUrl);
      }
      if (msg.asrJsSource) {
        const asrBlob = new Blob([msg.asrJsSource], { type: 'application/javascript' });
        const asrUrl = URL.createObjectURL(asrBlob);
        importScripts(asrUrl);
        URL.revokeObjectURL(asrUrl);
      }

      // 4. Compile WASM module.
      const wasmBytes = new Uint8Array(msg.wasmBinary);
      Module = await SherpaOnnx({
        wasmBinary: wasmBytes,
        print: (text) => self.postMessage({ type: 'log', message: text }),
        printErr: (text) => self.postMessage({ type: 'log', message: '[stderr] ' + text }),
      });

      // 5. Write model files to WASM FS.
      for (const [path, bytes] of Object.entries(msg.vadModelFiles || {})) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        writeFile(path, new Uint8Array(bytes));
      }
      for (const [path, bytes] of Object.entries(msg.asrModelFiles || {})) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        writeFile(path, new Uint8Array(bytes));
      }

      // 6. Create VAD instance.
      const vadConfig = msg.vadConfig;
      vad = createVad(Module, {
        sileroVad: vadConfig.sileroVad || { model: '', threshold: 0.5, minSilenceDuration: 0.5, minSpeechDuration: 0.25, windowSize: 512, maxSpeechDuration: 5.0 },
        tenVad: vadConfig.tenVad || { model: '', threshold: 0.5, minSilenceDuration: 0.5, minSpeechDuration: 0.25, windowSize: 256, maxSpeechDuration: 5.0 },
        sampleRate: vadConfig.sampleRate || 16000,
        numThreads: vadConfig.numThreads || 1,
        provider: vadConfig.provider || 'cpu',
        debug: vadConfig.debug || 1,
        bufferSizeInSeconds: 30,
      });

      // 7. Create ASR recognizer.
      recognizer = new OfflineRecognizer(msg.asrConfig, Module);

      self.postMessage({ type: 'ready' });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'acceptWaveform') {
    try {
      const samples = new Float32Array(msg.samples);
      const sampleRate = 16000;

      // Accept waveform into VAD.
      const pointer = Module._malloc(samples.length * 4);
      Module.HEAPF32.set(samples, pointer / 4);
      Module._SherpaOnnxVoiceActivityDetectorAcceptWaveform(
          vad.handle, pointer, samples.length);
      Module._free(pointer);

      // Report speech state.
      const isSpeaking = Module._SherpaOnnxVoiceActivityDetectorDetected(vad.handle) === 1;
      self.postMessage({ type: 'speechStateChanged', isSpeaking });

      // Process any completed segments (ASR runs inline).
      processDetectedSegments();
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'reset') {
    if (vad) {
      Module._SherpaOnnxVoiceActivityDetectorReset(vad.handle);
    }
    segmentIndex = 0;
  }

  else if (msg.type === 'dispose') {
    if (vad) { vad.free(); vad = null; }
    if (recognizer) { recognizer.free(); recognizer = null; }
    self.close();
  }
};
