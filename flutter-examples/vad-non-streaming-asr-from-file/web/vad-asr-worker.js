// VAD + ASR Web Worker — runs WASM VAD and ASR off the main thread.
//
// Messages: Main Thread → Worker
//
//   init — Initialize WASM module, create VAD and ASR instances.
//     {
//       type:          'init',
//       jsGlueSource:  String,       // sherpa-onnx-wasm-web.js
//       vadJsSource:   String,       // sherpa-onnx-vad.js
//       asrJsSource:   String,       // sherpa-onnx-asr.js
//       wasmBinary:    ArrayBuffer,  // compiled .wasm module
//       vadModelFiles: Object,       // { "relative/path": ArrayBuffer, ... }
//       asrModelFiles: Object,       // { "relative/path": ArrayBuffer, ... }
//       vadConfig:     Object,       // VadModelConfig JSON
//       asrConfig:     Object,       // OfflineRecognizerConfig JSON
//     }
//
//   runVad — Run VAD+ASR on audio samples.
//     {
//       type:                  'runVad',
//       runId:                 Number,       // run identifier for filtering stale messages
//       samples:               ArrayBuffer,  // Float32 PCM
//       sampleRate:            Number,
//       threshold:             Number,
//       minSilenceDuration:    Number,
//       minSpeechDuration:     Number,
//       maxSpeechDuration:     Number,
//     }
//
//   cancel — Cancel current processing.
//     { type: 'cancel' }
//
//   dispose — Destroy instances and close worker.
//     { type: 'dispose' }
//
// Messages: Worker → Main Thread
//
//   ready — Initialization complete.
//     { type: 'ready' }
//
//   started — A new runVad has started processing.
//     { type: 'started', runId: Number }
//
//   segment — A single segment with ASR text (sent as soon as ASR finishes it).
//     { type: 'segment', runId: Number, start: Number, end: Number, samples: Float32Array, text: String }
//
//   progress — Processing progress (0.0–1.0).
//     { type: 'progress', runId: Number, progress: Number }
//
//   result — Final VAD+ASR result with all segments.
//     {
//       type:          'result',
//       runId:         Number,
//       segments:      [{ start, end, samples, text }, ...],
//       elapsed:       Number,
//       audioDuration: Number,
//     }
//
//   log — Debug message.
//     { type: 'log', message: String }
//
//   error — Error message.
//     { type: 'error', message: String }

let Module = null;
let vad = null;
let recognizer = null;
let savedVadConfig = null;
let savedAsrConfig = null;
let _cancelled = false;
let _runId = 0;

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

self.onmessage = async function(e) {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      // 1. Load Emscripten JS glue.
      if (msg.jsGlueSource) {
        self.eval(msg.jsGlueSource);
      }

      // 2. Define module stub.
      self.eval('if (typeof module === "undefined") { var module = {}; }');

      // 3. Load VAD and ASR JS helpers via importScripts (Blob URLs).
      if (msg.vadJsSource) {
        const vadBlob = new Blob([msg.vadJsSource], { type: 'application/javascript' });
        const vadUrl = URL.createObjectURL(vadBlob);
        importScripts(vadUrl);
        URL.revokeObjectURL(vadUrl);
      }
      if (msg.asrJsSource) {
        // importScripts via Blob URL puts class declarations in global scope.
        const asrBlob = new Blob([msg.asrJsSource], { type: 'application/javascript' });
        const asrUrl = URL.createObjectURL(asrBlob);
        try {
          importScripts(asrUrl);
          self.postMessage({ type: 'log', message: 'ASR JS loaded, typeof OfflineRecognizer: ' + typeof OfflineRecognizer });
        } catch (e) {
          self.postMessage({ type: 'log', message: 'ASR JS load ERROR: ' + e.message });
        }
        URL.revokeObjectURL(asrUrl);
      }

      // 4. Compile WASM module.
      const wasmBytes = new Uint8Array(msg.wasmBinary);
      Module = await SherpaOnnx({
        wasmBinary: wasmBytes,
        print: (text) => self.postMessage({ type: 'log', message: text }),
        printErr: (text) => self.postMessage({ type: 'log', message: '[stderr] ' + text }),
      });

      // 5. Write VAD model files to WASM FS.
      for (const [path, bytes] of Object.entries(msg.vadModelFiles || {})) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        const arr = new Uint8Array(bytes);
        writeFile(path, arr);
        self.postMessage({ type: 'log', message: 'Wrote VAD: ' + path + ' (' + arr.length + ' bytes)' });
      }

      // 6. Write ASR model files to WASM FS.
      for (const [path, bytes] of Object.entries(msg.asrModelFiles || {})) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        const arr = new Uint8Array(bytes);
        writeFile(path, arr);
        self.postMessage({ type: 'log', message: 'Wrote ASR: ' + path + ' (' + arr.length + ' bytes)' });
      }

      // 7. Create VAD instance.
      savedVadConfig = msg.vadConfig;
      const vadConfig = {
        sileroVad: savedVadConfig.sileroVad || {
          model: '', threshold: 0.5, minSilenceDuration: 0.5,
          minSpeechDuration: 0.25, windowSize: 512, maxSpeechDuration: 5.0,
        },
        tenVad: savedVadConfig.tenVad || {
          model: '', threshold: 0.5, minSilenceDuration: 0.5,
          minSpeechDuration: 0.25, windowSize: 256, maxSpeechDuration: 5.0,
        },
        sampleRate: savedVadConfig.sampleRate || 16000,
        numThreads: savedVadConfig.numThreads || 1,
        provider: savedVadConfig.provider || 'cpu',
        debug: savedVadConfig.debug || 1,
        bufferSizeInSeconds: 30,
      };
      vad = createVad(Module, vadConfig);

      // 8. Create ASR OfflineRecognizer instance.
      savedAsrConfig = msg.asrConfig;
      recognizer = new OfflineRecognizer(savedAsrConfig, Module);

      self.postMessage({ type: 'ready' });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'cancel') {
    _cancelled = true;
  }

  else if (msg.type === 'runVad' && savedVadConfig) {
    try {
      _cancelled = false;
      _runId = msg.runId || 0;
      self.postMessage({ type: 'started', runId: _runId });

      // Re-create VAD with the requested parameters.
      if (vad) vad.free();

      const isTenVad = savedVadConfig.tenVad && savedVadConfig.tenVad.model;
      const userParams = {
        threshold: msg.threshold || 0.5,
        minSilenceDuration: msg.minSilenceDuration || 0.5,
        minSpeechDuration: msg.minSpeechDuration || 0.25,
        maxSpeechDuration: msg.maxSpeechDuration || 5.0,
      };

      const runConfig = {
        sileroVad: isTenVad
            ? (savedVadConfig.sileroVad || { model: '' })
            : { model: savedVadConfig.sileroVad.model, ...userParams,
                windowSize: savedVadConfig.sileroVad.windowSize || 512 },
        tenVad: isTenVad
            ? { model: savedVadConfig.tenVad.model, ...userParams,
                windowSize: savedVadConfig.tenVad.windowSize || 256 }
            : (savedVadConfig.tenVad || { model: '' }),
        sampleRate: savedVadConfig.sampleRate || 16000,
        numThreads: savedVadConfig.numThreads || 1,
        provider: savedVadConfig.provider || 'cpu',
        debug: savedVadConfig.debug || 1,
        bufferSizeInSeconds: 30,
      };
      vad = createVad(Module, runConfig);

      const startTime = performance.now();
      const samples = new Float32Array(msg.samples);
      const sampleRate = msg.sampleRate || 16000;
      const windowSize = isTenVad
          ? (savedVadConfig.tenVad.windowSize || 256)
          : (savedVadConfig.sileroVad.windowSize || 512);
      const numSamples = samples.length;
      const numIter = Math.floor(numSamples / windowSize);
      const audioDuration = numSamples / sampleRate;

      const segments = [];

      // Process VAD and run ASR on each segment immediately.
      function processDetectedSegments() {
        while (Module._SherpaOnnxVoiceActivityDetectorEmpty(vad.handle) !== 1) {
          const seg = extractSegment(vad.handle, sampleRate);

          // Run ASR on this segment right away.
          const stream = recognizer.createStream();
          stream.acceptWaveform(sampleRate, seg.samples);
          recognizer.decode(stream);
          const result = recognizer.getResult(stream);
          stream.free();
          seg.text = result.text || '';

          segments.push(seg);

          // Send to UI immediately.
          self.postMessage({
            type: 'segment',
            runId: _runId,
            start: seg.start,
            end: seg.end,
            samples: seg.samples,
            text: seg.text,
          });

          Module._SherpaOnnxVoiceActivityDetectorPop(vad.handle);
        }
      }

      for (let i = 0; i < numIter; i++) {
        if (_cancelled) break;

        const start = i * windowSize;
        const chunk = samples.slice(start, start + windowSize);

        // Accept waveform chunk.
        const pointer = Module._malloc(chunk.length * 4);
        Module.HEAPF32.set(chunk, pointer / 4);
        Module._SherpaOnnxVoiceActivityDetectorAcceptWaveform(
            vad.handle, pointer, chunk.length);
        Module._free(pointer);

        // Detect and process segments immediately.
        if (Module._SherpaOnnxVoiceActivityDetectorDetected(vad.handle) === 1) {
          processDetectedSegments();
        }

        self.postMessage({ type: 'progress', runId: _runId, progress: (i + 1) / numIter });
      }

      // Flush remaining segments.
      if (!_cancelled) {
        Module._SherpaOnnxVoiceActivityDetectorFlush(vad.handle);
        processDetectedSegments();
      }

      const elapsed = (performance.now() - startTime) / 1000;

      self.postMessage({
        type: 'result',
        runId: _runId,
        segments: segments,
        elapsed: elapsed,
        audioDuration: audioDuration,
      });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'dispose') {
    if (vad) { vad.free(); vad = null; }
    if (recognizer) { recognizer.free(); recognizer = null; }
    self.close();
  }
};

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
    text: '',
  };
}
