// TTS Web Worker — runs WASM generation off the main thread.
//
// Config helpers (initSherpaOnnxOfflineTtsConfig, freeConfig,
// initSherpaOnnxGenerationConfig, freeSherpaOnnxGenerationConfig)
// are loaded from sherpa-onnx-tts.js via eval at init time.
//
// ── Messages: Main Thread → Worker ─────────────────────────────────────────
//
// init — Initialize the WASM module and create a TTS instance.
//   {
//     type:          'init',
//     jsGlueSource:  String,       // sherpa-onnx-wasm-web.js (defines SherpaOnnx factory)
//     ttsJsSource:   String,       // sherpa-onnx-tts.js (config/generation helpers)
//     wasmBinary:    ArrayBuffer,  // compiled .wasm module
//     modelFiles:    Object,       // { "relative/path": ArrayBuffer, ... }
//     config:        Object,       // OfflineTtsConfig JSON (from toJson())
//   }
//
// generate — Synthesize speech from text.
//   {
//     type:                 'generate',
//     text:                 String,     // text to synthesize
//     sid:                  Number,     // speaker ID (default 0)
//     speed:                Number,     // speech rate (default 1.0)
//     generationId:         Number,     // id for matching chunks/done (default 0)
//     referenceAudio:       ArrayBuffer, // optional: Float32 PCM samples for voice cloning
//     referenceSampleRate:  Number,     // sample rate of reference audio
//     numSteps:             Number,     // diffusion steps (default 5)
//   }
//
// cancel — Abort the current generation.
//   { type: 'cancel' }
//
// dispose — Destroy the TTS instance and close the worker.
//   { type: 'dispose' }
//
// ── Messages: Worker → Main Thread ─────────────────────────────────────────
//
// ready — TTS initialized successfully.
//   { type: 'ready', numSpeakers: Number, sampleRate: Number }
//
// chunk — Streaming audio chunk (sent during generation).
//   {
//     type:         'chunk',
//     samples:      ArrayBuffer,  // Float32 PCM (transferred, not copied)
//     progress:     Number,       // 0.0–1.0
//     sampleRate:   Number,
//     generationId: Number,
//   }
//
// done — Generation complete.
//   {
//     type:         'done',
//     samples:      ArrayBuffer,  // Float32 PCM (transferred)
//     sampleRate:   Number,
//     duration:     Number,       // audio duration in seconds
//     elapsed:      Number,       // wall-clock time in seconds
//     generationId: Number,
//   }
//
// log — Debug/info message from the WASM module (stdout/stderr).
//   { type: 'log', message: String }
//
// error — Error message.
//   { type: 'error', message: String }

let Module = null;
let tts = null;
let _cancelled = false;

// ── Emscripten FS helpers ────────────────────────────────────────────────

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

// ── Message handler ──────────────────────────────────────────────────────

self.onmessage = async function(e) {
  const msg = e.data;

  if (msg.type === 'init') {
    try {
      // 1. Load Emscripten JS glue (defines SherpaOnnx factory).
      if (msg.jsGlueSource) {
        self.eval(msg.jsGlueSource);
      }

      // 2. Load sherpa-onnx-tts.js helpers (defines initSherpaOnnxOfflineTtsConfig,
      //    freeConfig, initSherpaOnnxGenerationConfig, freeSherpaOnnxGenerationConfig, etc.)
      if (msg.ttsJsSource) {
        self.eval(msg.ttsJsSource);
      }

      // 3. Compile WASM module.
      const wasmBytes = new Uint8Array(msg.wasmBinary);
      Module = await SherpaOnnx({
        wasmBinary: wasmBytes,
        print: (text) => self.postMessage({ type: 'log', message: text }),
        printErr: (text) => self.postMessage({ type: 'log', message: '[stderr] ' + text }),
      });

      // 4. Write model files to WASM FS.
      const modelFiles = msg.modelFiles;
      for (const [path, bytes] of Object.entries(modelFiles)) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        writeFile(path, new Uint8Array(bytes));
      }

      // 5. Create TTS instance using sherpa-onnx-tts.js helper.
      const config = initSherpaOnnxOfflineTtsConfig(msg.config, Module);
      const handle = Module._SherpaOnnxCreateOfflineTts(config.ptr);
      freeConfig(config, Module);

      if (!handle) {
        self.postMessage({ type: 'error', message: 'Failed to create TTS (null handle)' });
        return;
      }

      const sampleRate = Module._SherpaOnnxOfflineTtsSampleRate(handle);
      const numSpeakers = Module._SherpaOnnxOfflineTtsNumSpeakers(handle);

      tts = { handle, sampleRate, numSpeakers };
      self.postMessage({ type: 'ready', numSpeakers, sampleRate });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'generate' && tts) {
    try {
      const startTime = performance.now();

      const genCfg = {
        silenceScale: 0.2,
        speed: msg.speed || 1.0,
        sid: msg.sid || 0,
      };

      // Reference audio for voice cloning (e.g. Pocket TTS).
      if (msg.referenceAudio) {
        genCfg.referenceAudio = new Float32Array(msg.referenceAudio);
        genCfg.referenceSampleRate = msg.referenceSampleRate || 0;
        genCfg.numSteps = msg.numSteps || 5;
      }

      const cfgWasm = initSherpaOnnxGenerationConfig(genCfg, Module);

      // Set up callback for streaming chunks.
      _cancelled = false;
      const genId = msg.generationId || 0;
      const callbackPtr = Module.addFunction((samplesPtr, n, progress, arg) => {
        if (_cancelled) return 0;
        const samples = new Float32Array(Module.HEAPF32.buffer, samplesPtr, n).slice();
        self.postMessage({
          type: 'chunk',
          samples: samples.buffer,
          progress: progress,
          sampleRate: tts.sampleRate,
          generationId: genId,
        }, [samples.buffer]);
        return 1;
      }, 'iiifi');

      // Prepare text.
      const textLen = Module.lengthBytesUTF8(msg.text) + 1;
      const textPtr = Module._malloc(textLen);
      Module.stringToUTF8(msg.text, textPtr, textLen);

      // Generate.
      const audioPtr = Module._SherpaOnnxOfflineTtsGenerateWithConfig(
        tts.handle, textPtr, cfgWasm.ptr, callbackPtr, 0);

      Module._free(textPtr);
      freeSherpaOnnxGenerationConfig(cfgWasm, Module);
      Module.removeFunction(callbackPtr);

      if (!audioPtr) {
        self.postMessage({ type: 'error', message: 'Generation failed' });
        return;
      }

      // Read result.
      const base = audioPtr / 4;
      const samplesPtr = Module.HEAPU32[base];
      const numSamples = Module.HEAP32[base + 1];
      const sampleRateOut = Module.HEAP32[base + 2];

      const samples = new Float32Array(Module.HEAPF32.buffer, samplesPtr, numSamples).slice();

      Module._SherpaOnnxDestroyOfflineTtsGeneratedAudio(audioPtr);

      const elapsed = (performance.now() - startTime) / 1000;
      const duration = numSamples / sampleRateOut;

      self.postMessage({
        type: 'done',
        samples: samples.buffer,
        sampleRate: sampleRateOut,
        duration: duration,
        elapsed: elapsed,
        generationId: genId,
      }, [samples.buffer]);
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'cancel') {
    _cancelled = true;
  }

  else if (msg.type === 'dispose') {
    if (tts) {
      Module._SherpaOnnxDestroyOfflineTts(tts.handle);
      tts = null;
    }
    self.close();
  }
};
