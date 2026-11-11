// VAD Web Worker — runs WASM VAD for real-time microphone input.
//
// Messages: Main Thread → Worker
//
//   init — Initialize WASM module and create VAD instance.
//     {
//       type:          'init',
//       jsGlueSource:  String,
//       vadJsSource:   String,
//       wasmBinary:    ArrayBuffer,
//       modelFiles:    Object,
//       config:        Object,
//     }
//
//   acceptWaveform — Send audio samples to VAD.
//     { type: 'acceptWaveform', samples: ArrayBuffer }
//
//   dispose — Destroy instance and close worker.
//     { type: 'dispose' }
//
// Messages: Worker → Main Thread
//
//   ready — Initialization complete.
//     { type: 'ready' }
//
//   speechStateChanged — Speech detected or ended.
//     { type: 'speechStateChanged', isSpeaking: Boolean }
//
//   segmentCountChanged — New segment detected.
//     { type: 'segmentCountChanged', count: Number }
//
//   log — Debug message.
//     { type: 'log', message: String }
//
//   error — Error message.
//     { type: 'error', message: String }

let Module = null;
let vad = null;
let segmentCount = 0;
let isSpeaking = false;

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
      if (msg.jsGlueSource) {
        self.eval(msg.jsGlueSource);
      }

      self.eval('if (typeof module === "undefined") { var module = {}; }');

      if (msg.vadJsSource) {
        self.eval(msg.vadJsSource);
      }

      const wasmBytes = new Uint8Array(msg.wasmBinary);
      Module = await SherpaOnnx({
        wasmBinary: wasmBytes,
        print: (text) => self.postMessage({ type: 'log', message: text }),
        printErr: (text) => self.postMessage({ type: 'log', message: '[stderr] ' + text }),
      });

      const modelFiles = msg.modelFiles;
      for (const [path, bytes] of Object.entries(modelFiles)) {
        const dir = path.substring(0, path.lastIndexOf('/'));
        if (dir) mkdirTree(dir);
        writeFile(path, new Uint8Array(bytes));
      }

      const config = msg.config;
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
        bufferSizeInSeconds: 30,
      };

      vad = createVad(Module, vadConfig);
      segmentCount = 0;
      isSpeaking = false;

      self.postMessage({ type: 'ready' });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'acceptWaveform' && vad) {
    try {
      const samples = new Float32Array(msg.samples);

      const pointer = Module._malloc(samples.length * 4);
      Module.HEAPF32.set(samples, pointer / 4);
      Module._SherpaOnnxVoiceActivityDetectorAcceptWaveform(
          vad.handle, pointer, samples.length);
      Module._free(pointer);

      const detected = Module._SherpaOnnxVoiceActivityDetectorDetected(vad.handle) === 1;
      if (detected !== isSpeaking) {
        isSpeaking = detected;
        self.postMessage({ type: 'speechStateChanged', isSpeaking: isSpeaking });
      }

      // Collect completed segments (available after speech ends).
      while (Module._SherpaOnnxVoiceActivityDetectorEmpty(vad.handle) !== 1) {
        const h = Module._SherpaOnnxVoiceActivityDetectorFront(vad.handle);
        // SherpaOnnxSpeechSegment (wasm32): int64 start @0, float *samples @8, int32 n @12
        // Rebuild the 64-bit `start` from its two little-endian 32-bit heap halves:
        //   value = high * 2^32 + low, where 2^32 = 4294967296 = 0x100000000.
        // `>>> 0` reads the low half unsigned (HEAP32 is signed).
        // A JS Number is an IEEE754 double and holds integers exactly up to
        // 2^53, i.e. ~17,800 years of 16 kHz samples, so no BigInt is needed.
        const segStartLo = Module.HEAP32[h / 4] >>> 0;
        const segStartHi = Module.HEAP32[h / 4 + 1];
        const segStart = segStartHi * 4294967296 /* 2^32 */ + segStartLo;
        const samplesPtr = Module.HEAP32[h / 4 + 2] / 4;
        const numSegSamples = Module.HEAP32[h / 4 + 3];

        const segSamples = new Float32Array(numSegSamples);
        for (let j = 0; j < numSegSamples; j++) {
          segSamples[j] = Module.HEAPF32[samplesPtr + j];
        }

        Module._SherpaOnnxDestroySpeechSegment(h);

        const segStartTime = segStart / 16000;
        const segEndTime = segStartTime + numSegSamples / 16000;

        self.postMessage({
          type: 'segmentDetected',
          start: segStartTime,
          end: segEndTime,
          samples: segSamples.buffer,
        }, [segSamples.buffer]);

        Module._SherpaOnnxVoiceActivityDetectorPop(vad.handle);
        segmentCount++;
      }
      self.postMessage({ type: 'segmentCountChanged', count: segmentCount });
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
