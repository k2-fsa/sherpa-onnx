// Punctuation Web Worker — runs WASM punctuation off the main thread.
//
// Messages: Main Thread → Worker
//
//   init — Initialize WASM module and create punctuation instance.
//     {
//       type:          'init',
//       jsGlueSource:  String,       // sherpa-onnx-wasm-web.js
//       punctJsSource: String,       // sherpa-onnx-punctuation.js
//       wasmBinary:    ArrayBuffer,  // compiled .wasm module
//       modelFiles:    Object,       // { "relative/path": ArrayBuffer, ... }
//       config:        Object,       // OnlinePunctuationConfig JSON
//     }
//
//   punctuate — Add punctuation to text.
//     { type: 'punctuate', text: String }
//
//   dispose — Destroy instance and close worker.
//     { type: 'dispose' }
//
// Messages: Worker → Main Thread
//
//   ready — Initialization complete.
//     { type: 'ready' }
//
//   result — Punctuated text.
//     { type: 'result', result: String, elapsed: Number }
//
//   log — Debug message.
//     { type: 'log', message: String }
//
//   error — Error message.
//     { type: 'error', message: String }

let Module = null;
let punct = null;

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

      // 3. Load punctuation JS helpers.
      if (msg.punctJsSource) {
        self.eval(msg.punctJsSource);
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
        const arr = new Uint8Array(bytes);
        writeFile(path, arr);
        self.postMessage({ type: 'log', message: 'Wrote ' + path + ' (' + arr.length + ' bytes)' });
      }

      // 5. Create punctuation instance (online).
      const config = initSherpaOnnxOnlinePunctuationConfig(msg.config, Module);
      punct = {
        handle: Module._SherpaOnnxCreateOnlinePunctuation(config.ptr),
      };
      freeConfig(config, Module);

      if (!punct.handle) {
        self.postMessage({ type: 'error', message: 'Failed to create punctuation (null handle)' });
        return;
      }

      self.postMessage({ type: 'ready' });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'punctuate' && punct) {
    try {
      const startTime = performance.now();

      const textLen = Module.lengthBytesUTF8(msg.text) + 1;
      const textPtr = Module._malloc(textLen);
      Module.stringToUTF8(msg.text, textPtr, textLen);

      const resultPtr = Module._SherpaOnnxOnlinePunctuationAddPunct(punct.handle, textPtr);
      const result = resultPtr ? Module.UTF8ToString(resultPtr) : '';

      if (resultPtr) {
        Module._SherpaOnnxOnlinePunctuationFreeText(resultPtr);
      }
      Module._free(textPtr);

      const elapsed = (performance.now() - startTime) / 1000;

      self.postMessage({ type: 'result', result: result, elapsed: elapsed });
    } catch (e) {
      self.postMessage({ type: 'error', message: e.message || String(e) });
    }
  }

  else if (msg.type === 'dispose') {
    if (punct) {
      Module._SherpaOnnxDestroyOnlinePunctuation(punct.handle);
      punct = null;
    }
    self.close();
  }
};
