// VAD Web Worker — runs WASM VAD off the main thread.
//
// Messages: Main Thread → Worker
//
//   init — Initialize WASM module and create VAD instance.
//     {
//       type:          'init',
//       jsGlueSource:  String,       // sherpa-onnx-wasm-web.js
//       vadJsSource:   String,       // sherpa-onnx-vad.js
//       wasmBinary:    ArrayBuffer,  // compiled .wasm module
//       modelFiles:    Object,       // { "relative/path": ArrayBuffer, ... }
//       config:        Object,       // VadModelConfig JSON
//     }
//
//   runVad — Run VAD on audio samples.
//     {
//       type:                  'runVad',
//       samples:               ArrayBuffer,  // Float32 PCM
//       sampleRate:            Number,
//       threshold:             Number,
//       minSilenceDuration:    Number,
//       minSpeechDuration:     Number,
//       maxSpeechDuration:     Number,
//     }
//
//   dispose — Destroy instance and close worker.
//     { type: 'dispose' }
//
// Messages: Worker → Main Thread
//
//   ready — Initialization complete.
//     { type: 'ready' }
//
//   progress — Processing progress.
//     { type: 'progress', progress: Number }
//
//   result — VAD result with segments.
//     {
//       type:          'result',
//       segments:      [{ start: Number, end: Number, samples: Float32Array }, ...],
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
let savedConfig = null;
let _cancelled = false;

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

      // 2. Define module stub (Node.js pattern used by JS wrappers).
      self.eval('if (typeof module === "undefined") { var module = {}; }');

      // 3. Load VAD JS helpers.
      if (msg.vadJsSource) {
        self.eval(msg.vadJsSource);
      }

      // 4. Compile WASM module.
      const wasmBytes = new Uint8Array(msg.wasmBinary);
      Module = await SherpaOnnx({
        wasmBinary: wasmBytes,
        print: (text) => self.postMessage({ type: 'log', message: text }),
        printErr: (text) => self.postMessage({ type: 'log', message: '[stderr] ' + text }),
      });

      // 5. Write model files to WASM FS.
      const modelFiles = msg.modelFiles;
      for (const [path, bytes] of Object.entries(modelFiles)) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        const arr = new Uint8Array(bytes);
        writeFile(path, arr);
        self.postMessage({ type: 'log', message: 'Wrote ' + path + ' (' + arr.length + ' bytes)' });
      }

      // 6. Create VAD instance.
      const config = msg.config;
      savedConfig = config;
      const bufferSizeInSeconds = 30;

      // Build config for createVad.
      const vadConfig = {
        sileroVad: config.sileroVad || {
          model: '',
          threshold: 0.5,
          minSilenceDuration: 0.5,
          minSpeechDuration: 0.25,
          windowSize: 512,
          maxSpeechDuration: 5.0,
        },
        tenVad: config.tenVad || {
          model: '',
          threshold: 0.5,
          minSilenceDuration: 0.5,
          minSpeechDuration: 0.25,
          windowSize: 256,
          maxSpeechDuration: 5.0,
        },
        sampleRate: config.sampleRate || 16000,
        numThreads: config.numThreads || 1,
        provider: config.provider || 'cpu',
        debug: config.debug || 1,
        bufferSizeInSeconds: bufferSizeInSeconds,
      };

      vad = createVad(Module, vadConfig);

      self.postMessage({ type: 'ready' });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'cancel') {
    _cancelled = true;
  }

  else if (msg.type === 'runVad' && savedConfig) {
    try {
      _cancelled = false;

      // Re-create VAD with the requested parameters.
      if (vad) {
        vad.free();
      }

      // Determine which model is active.
      const isTenVad = savedConfig.tenVad && savedConfig.tenVad.model;
      const userParams = {
        threshold: msg.threshold || 0.5,
        minSilenceDuration: msg.minSilenceDuration || 0.5,
        minSpeechDuration: msg.minSpeechDuration || 0.25,
        maxSpeechDuration: msg.maxSpeechDuration || 5.0,
      };

      const runConfig = {
        sileroVad: isTenVad
            ? (savedConfig.sileroVad || { model: '' })
            : {
                model: savedConfig.sileroVad.model,
                ...userParams,
                windowSize: savedConfig.sileroVad.windowSize || 512,
              },
        tenVad: isTenVad
            ? {
                model: savedConfig.tenVad.model,
                ...userParams,
                windowSize: savedConfig.tenVad.windowSize || 256,
              }
            : (savedConfig.tenVad || { model: '' }),
        sampleRate: savedConfig.sampleRate || 16000,
        numThreads: savedConfig.numThreads || 1,
        provider: savedConfig.provider || 'cpu',
        debug: savedConfig.debug || 1,
        bufferSizeInSeconds: 30,
      };
      vad = createVad(Module, runConfig);

      const startTime = performance.now();
      const samples = new Float32Array(msg.samples);
      const sampleRate = msg.sampleRate || 16000;
      const windowSize = isTenVad
          ? (savedConfig.tenVad.windowSize || 256)
          : (savedConfig.sileroVad.windowSize || 512);
      const numSamples = samples.length;
      const numIter = Math.floor(numSamples / windowSize);
      const audioDuration = numSamples / sampleRate;

      const segments = [];

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

        // Check for detected speech segments.
        if (Module._SherpaOnnxVoiceActivityDetectorDetected(vad.handle) === 1) {
          while (Module._SherpaOnnxVoiceActivityDetectorEmpty(vad.handle) !== 1) {
            const h = Module._SherpaOnnxVoiceActivityDetectorFront(vad.handle);
            const segStart = Module.HEAP32[h / 4];
            const samplesPtr = Module.HEAP32[h / 4 + 1] / 4;
            const numSegSamples = Module.HEAP32[h / 4 + 2];

            const segSamples = new Float32Array(numSegSamples);
            for (let j = 0; j < numSegSamples; j++) {
              segSamples[j] = Module.HEAPF32[samplesPtr + j];
            }

            Module._SherpaOnnxDestroySpeechSegment(h);

            const segStartTime = segStart / sampleRate;
            const segEndTime = segStartTime + numSegSamples / sampleRate;
            segments.push({
              start: segStartTime,
              end: segEndTime,
              samples: segSamples,
            });

            Module._SherpaOnnxVoiceActivityDetectorPop(vad.handle);
          }
        }

        // Report progress.
        const progress = (i + 1) / numIter;
        self.postMessage({ type: 'progress', progress: progress });
      }

      // Flush remaining segments.
      Module._SherpaOnnxVoiceActivityDetectorFlush(vad.handle);
      while (Module._SherpaOnnxVoiceActivityDetectorEmpty(vad.handle) !== 1) {
        const h = Module._SherpaOnnxVoiceActivityDetectorFront(vad.handle);
        const segStart = Module.HEAP32[h / 4];
        const samplesPtr = Module.HEAP32[h / 4 + 1] / 4;
        const numSegSamples = Module.HEAP32[h / 4 + 2];

        const segSamples = new Float32Array(numSegSamples);
        for (let j = 0; j < numSegSamples; j++) {
          segSamples[j] = Module.HEAPF32[samplesPtr + j];
        }

        Module._SherpaOnnxDestroySpeechSegment(h);

        const segStartTime = segStart / sampleRate;
        const segEndTime = segStartTime + numSegSamples / sampleRate;
        segments.push({
          start: segStartTime,
          end: segEndTime,
          samples: segSamples,
        });

        Module._SherpaOnnxVoiceActivityDetectorPop(vad.handle);
      }

      const elapsed = (performance.now() - startTime) / 1000;

      self.postMessage({
        type: 'result',
        segments: segments,
        elapsed: elapsed,
        audioDuration: audioDuration,
      });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'dispose') {
    if (vad) {
      vad.free();
      vad = null;
    }
    self.close();
  }
};
