let tts = null;
self.Module = {
  // https://emscripten.org/docs/api_reference/module.html#Module.locateFile
  locateFile: function (path, scriptDirectory = "") {
    return scriptDirectory + path;
  },
  // https://emscripten.org/docs/api_reference/module.html#Module.locateFile
  setStatus: function (status) {
    self.postMessage({ type: "sherpa-onnx-tts-progress", status });
  },
  onRuntimeInitialized: function () {
    console.log("Model files downloaded!");
    console.log("Initializing tts ......");
    try {
      tts = createOfflineTts(self.Module);
      self.postMessage({
        type: "sherpa-onnx-tts-ready",
        numSpeakers: tts.numSpeakers,
      });
    } catch (e) {
      self.postMessage({
        type: "error",
        message: "TTS Initialization failed: " + e.message,
      });
    }
  },
};
importScripts("/sherpa-onnx-wasm-main-tts.js");
importScripts("/sherpa-onnx-tts.js");
self.onmessage = async (e) => {
  const { type, text, sid, speed } = e.data;
  if (type === "generate") {
    if (!tts) {
      return;
    }
    try {
      const audio = tts.generate({
        text: text,
        sid: sid || 0,
        speed: speed || 1.0,
      });
      const samples = audio.samples;
      const sampleRate = tts.sampleRate;
      self.postMessage(
        {
          type: "sherpa-onnx-tts-result",
          samples: samples,
          sampleRate: sampleRate,
        },
        [samples.buffer],
      );
    } catch (err) {
      self.postMessage({
        type: "error",
        message: "Generation failed: " + err.message,
      });
    }
  }
};
