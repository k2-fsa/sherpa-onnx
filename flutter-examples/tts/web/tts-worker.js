// TTS Web Worker — runs WASM generation off the main thread.
// Receives model bytes and config via postMessage, posts audio chunks back.

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

// ── TTS config helpers (matching wasm/tts/sherpa-onnx-tts.js) ──────────────
// Layout follows the C structs in sherpa-onnx/c-api/c-api.h exactly.

function freeConfig(config) {
  if (!config) return;
  if ('buffer' in config && config.buffer) Module._free(config.buffer);
  if ('config' in config) freeConfig(config.config);
  if ('modelCfg' in config) freeConfig(config.modelCfg);
  if ('matcha' in config) freeConfig(config.matcha);
  if ('kokoro' in config) freeConfig(config.kokoro);
  if ('kitten' in config) freeConfig(config.kitten);
  if ('zipvoice' in config) freeConfig(config.zipvoice);
  if ('pocket' in config) freeConfig(config.pocket);
  if ('supertonic' in config) freeConfig(config.supertonic);
  if (config.ptr) Module._free(config.ptr);
}

// SherpaOnnxOfflineTtsVitsModelConfig: 8 pointers/fields (8*4 = 32 bytes)
// Fields: model, lexicon, tokens, data_dir, noise_scale, noise_scale_w, length_scale, dict_dir
function initVitsConfig(cfg, M) {
  const modelLen = M.lengthBytesUTF8(cfg.model || '') + 1;
  const lexiconLen = M.lengthBytesUTF8(cfg.lexicon || '') + 1;
  const tokensLen = M.lengthBytesUTF8(cfg.tokens || '') + 1;
  const dataDirLen = M.lengthBytesUTF8(cfg.dataDir || '') + 1;
  const dictDir = '';
  const dictDirLen = M.lengthBytesUTF8(dictDir) + 1;
  const n = modelLen + lexiconLen + tokensLen + dataDirLen + dictDirLen;
  const buf = M._malloc(n);
  const len = 8 * 4;
  const ptr = M._malloc(len);
  let off = 0;
  M.stringToUTF8(cfg.model || '', buf + off, modelLen); off += modelLen;
  M.stringToUTF8(cfg.lexicon || '', buf + off, lexiconLen); off += lexiconLen;
  M.stringToUTF8(cfg.tokens || '', buf + off, tokensLen); off += tokensLen;
  M.stringToUTF8(cfg.dataDir || '', buf + off, dataDirLen); off += dataDirLen;
  M.stringToUTF8(dictDir, buf + off, dictDirLen); off += dictDirLen;
  off = 0;
  M.setValue(ptr, buf + off, 'i8*'); off += modelLen;
  M.setValue(ptr + 4, buf + off, 'i8*'); off += lexiconLen;
  M.setValue(ptr + 8, buf + off, 'i8*'); off += tokensLen;
  M.setValue(ptr + 12, buf + off, 'i8*'); off += dataDirLen;
  M.setValue(ptr + 16, cfg.noiseScale || 0.667, 'float');
  M.setValue(ptr + 20, cfg.noiseScaleW || 0.8, 'float');
  M.setValue(ptr + 24, cfg.lengthScale || 1.0, 'float');
  M.setValue(ptr + 28, buf + off, 'i8*'); off += dictDirLen;
  return { buffer: buf, ptr: ptr, len: len };
}

// SherpaOnnxOfflineTtsModelConfig layout (c-api.h:2409-2430):
//   vits, num_threads, debug, provider, matcha, kokoro, kitten, zipvoice, pocket, supertonic
function initModelConfig(cfg, M) {
  // Support key aliases: Dart toJson() uses short names.
  if ('vits' in cfg && !('offlineTtsVitsModelConfig' in cfg)) {
    cfg.offlineTtsVitsModelConfig = cfg.vits;
  }

  const vits = initVitsConfig(cfg.offlineTtsVitsModelConfig || {}, M);

  // Total size of SherpaOnnxOfflineTtsModelConfig:
  // vits(32) + num_threads(4) + debug(4) + provider(4) +
  // matcha(32) + kokoro(32) + kitten(20) + zipvoice(40) + pocket(32) + supertonic(28) = 228
  const len = 32 + 4 + 4 + 4 + 32 + 32 + 20 + 40 + 32 + 28;
  const ptr = M._malloc(len);
  // Zero the entire buffer first (unused model configs should be all zeros).
  M.HEAPU8.fill(0, ptr, ptr + len);

  let off = 0;
  // vits
  M._CopyHeap(vits.ptr, vits.len, ptr + off); off += 32;
  // num_threads, debug, provider
  M.setValue(ptr + off, cfg.numThreads || 1, 'i32'); off += 4;
  M.setValue(ptr + off, cfg.debug || 0, 'i32'); off += 4;
  const provLen = M.lengthBytesUTF8(cfg.provider || 'cpu') + 1;
  const provBuf = M._malloc(provLen);
  M.stringToUTF8(cfg.provider || 'cpu', provBuf, provLen);
  M.setValue(ptr + off, provBuf, 'i8*'); off += 4;
  // Remaining model configs are already zeroed — C++ treats empty strings as "not configured".

  return { buffer: provBuf, ptr: ptr, len: len, config: vits };
}

// SherpaOnnxOfflineTtsConfig layout (c-api.h:2450-2461):
//   model, rule_fsts, max_num_sentences, rule_fars, silence_scale
function initTtsConfig(cfg, M) {
  // Support key alias: Dart toJson() uses 'model', sherpa-onnx-tts.js uses 'offlineTtsModelConfig'.
  if ('model' in cfg && !('offlineTtsModelConfig' in cfg)) {
    cfg.offlineTtsModelConfig = cfg.model;
  }
  // Support typo in Dart config class.
  if ('maxNumSenetences' in cfg && !('maxNumSentences' in cfg)) {
    cfg.maxNumSentences = cfg.maxNumSenetences;
  }

  const modelCfg = initModelConfig(cfg.offlineTtsModelConfig || {}, M);
  const len = modelCfg.len + 4 * 4;
  const ptr = M._malloc(len);
  let off = 0;
  M._CopyHeap(modelCfg.ptr, modelCfg.len, ptr + off); off += modelCfg.len;

  const ruleFstsLen = M.lengthBytesUTF8(cfg.ruleFsts || '') + 1;
  const ruleFarsLen = M.lengthBytesUTF8(cfg.ruleFars || '') + 1;
  const buf = M._malloc(ruleFstsLen + ruleFarsLen);
  M.stringToUTF8(cfg.ruleFsts || '', buf, ruleFstsLen);
  M.stringToUTF8(cfg.ruleFars || '', buf + ruleFstsLen, ruleFarsLen);

  M.setValue(ptr + off, buf, 'i8*'); off += 4;
  M.setValue(ptr + off, cfg.maxNumSentences || 1, 'i32'); off += 4;
  M.setValue(ptr + off, buf + ruleFstsLen, 'i8*'); off += 4;
  M.setValue(ptr + off, cfg.silenceScale || 0.2, 'float');

  return { buffer: buf, ptr: ptr, len: len, modelCfg: modelCfg };
}

// ── Generation config (from sherpa-onnx-tts.js) ──────────────────────────

function initSherpaOnnxGenerationConfig(config, Module) {
  const len = 9 * 4;
  const ptr = Module._malloc(len);

  Module.setValue(ptr + 0 * 4, config.silenceScale || 0.2, 'float');
  Module.setValue(ptr + 1 * 4, config.speed || 1.0, 'float');
  Module.setValue(ptr + 2 * 4, config.sid || 0, 'i32');

  let referenceAudioPtr = 0;
  if (config.referenceAudio && config.referenceAudio.length > 0) {
    referenceAudioPtr = Module._malloc(config.referenceAudio.length * 4);
    Module.HEAPF32.set(config.referenceAudio, referenceAudioPtr / 4);
  }
  Module.setValue(ptr + 3 * 4, referenceAudioPtr, 'i8*');

  Module.setValue(
      ptr + 4 * 4, config.referenceAudio ? config.referenceAudio.length : 0,
      'i32');

  Module.setValue(ptr + 5 * 4, config.referenceSampleRate || 0, 'i32');

  let referenceTextPtr = 0;
  if (config.referenceText) {
    const textLen = Module.lengthBytesUTF8(config.referenceText) + 1;
    referenceTextPtr = Module._malloc(textLen);
    Module.stringToUTF8(config.referenceText, referenceTextPtr, textLen);
  }
  Module.setValue(ptr + 6 * 4, referenceTextPtr, 'i8*');

  Module.setValue(ptr + 7 * 4, config.numSteps || 5, 'i32');

  let extraPtr = 0;
  let extraStr = null;
  if (config.extra) {
    if (typeof config.extra === 'object') {
      extraStr = JSON.stringify(config.extra);
    } else if (typeof config.extra === 'string') {
      extraStr = config.extra;
    }
  }
  if (extraStr !== null) {
    const extraLen = Module.lengthBytesUTF8(extraStr) + 1;
    extraPtr = Module._malloc(extraLen);
    Module.stringToUTF8(extraStr, extraPtr, extraLen);
  }
  Module.setValue(ptr + 8 * 4, extraPtr, 'i8*');

  return { ptr, referenceAudioPtr, referenceTextPtr, extraPtr };
}

function freeSherpaOnnxGenerationConfig(cfg, Module) {
  if (!cfg) return;
  if (cfg.referenceAudioPtr) Module._free(cfg.referenceAudioPtr);
  if (cfg.referenceTextPtr) Module._free(cfg.referenceTextPtr);
  if (cfg.extraPtr) Module._free(cfg.extraPtr);
  if (cfg.ptr) Module._free(cfg.ptr);
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

      // 2. Compile WASM module.
      const wasmBytes = new Uint8Array(msg.wasmBinary);
      Module = await SherpaOnnx({
        wasmBinary: wasmBytes,
        print: (text) => self.postMessage({ type: 'log', message: text }),
        printErr: (text) => self.postMessage({ type: 'log', message: '[stderr] ' + text }),
      });

      // 2. Write model files to WASM FS.
      const modelFiles = msg.modelFiles;
      for (const [path, bytes] of Object.entries(modelFiles)) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        writeFile(path, new Uint8Array(bytes));
      }

      // 3. Create TTS instance.
      const ttsConfig = initTtsConfig(msg.config, Module);
      self.postMessage({ type: 'log', message: 'TTS config created, ptr=' + ttsConfig.ptr });

      // Verify key model files exist before calling into WASM.
      const cfg = msg.config.model || msg.config.offlineTtsModelConfig || {};
      const vitsCfg = cfg.vits || cfg.offlineTtsVitsModelConfig || {};
      const checkPaths = [
        vitsCfg.model, vitsCfg.tokens,
        vitsCfg.dataDir ? vitsCfg.dataDir + '/phontab' : '',
      ].filter(Boolean);
      for (const p of checkPaths) {
        try {
          const stat = getFS().stat(p);
          self.postMessage({ type: 'log', message: '  OK: ' + p + ' (' + stat.size + ' bytes)' });
        } catch (_) {
          self.postMessage({ type: 'log', message: '  MISSING: ' + p });
        }
      }

      let handle = 0;
      try {
        handle = Module._SherpaOnnxCreateOfflineTts(ttsConfig.ptr);
      } catch (e) {
        self.postMessage({ type: 'log', message: 'TTS create threw: ' + (e.message || e) });
      }
      self.postMessage({ type: 'log', message: 'TTS handle=' + handle });
      freeConfig(ttsConfig);

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

      // Build generation config (same layout as SherpaOnnxGenerationConfig in C).
      const genCfg = {
        silenceScale: 0.2,
        speed: msg.speed || 1.0,
        sid: msg.sid || 0,
      };
      const genCfgResult = initSherpaOnnxGenerationConfig(genCfg, Module);

      // Set up callback for streaming chunks.
      _cancelled = false;
      const genId = msg.generationId || 0;
      const callbackPtr = Module.addFunction((samplesPtr, n, progress, arg) => {
        if (_cancelled) return 0; // cancel generation
        const samples = new Float32Array(Module.HEAPF32.buffer, samplesPtr, n).slice();
        self.postMessage({
          type: 'chunk',
          samples: samples.buffer,
          progress: progress,
          sampleRate: tts.sampleRate,
          generationId: genId,
        }, [samples.buffer]);
        return 1; // continue
      }, 'iiifi');

      // Prepare text.
      const textLen = Module.lengthBytesUTF8(msg.text) + 1;
      const textPtr = Module._malloc(textLen);
      Module.stringToUTF8(msg.text, textPtr, textLen);

      // Generate.
      const audioPtr = Module._SherpaOnnxOfflineTtsGenerateWithConfig(
        tts.handle, textPtr, genCfgResult.ptr, callbackPtr, 0);

      Module._free(textPtr);
      freeSherpaOnnxGenerationConfig(genCfgResult, Module);
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
