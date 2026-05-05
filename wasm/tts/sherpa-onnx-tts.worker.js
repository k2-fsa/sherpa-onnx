// Module-type worker. The wasm glue is built with -sEXPORT_ES6=1 so its
// pthread runtime can spawn its own worker pool from inside this worker
// (classic-worker importScripts() of the same glue hangs during nested
// pthread bootstrap).
import createModule from "./sherpa-onnx-wasm-main-tts.js";
import {
  createOfflineTts,
  getDefaultOfflineTtsModelType,
} from "./sherpa-onnx-tts.js";

let tts = null;

const Module = await createModule({
  locateFile: (path, scriptDirectory = "") => scriptDirectory + path,
  setStatus: (status) => {
    self.postMessage({ type: "sherpa-onnx-tts-progress", status });
  },
});

try {
  tts = createOfflineTts(Module);
  self.postMessage({
    type: "sherpa-onnx-tts-ready",
    modelType: getDefaultOfflineTtsModelType(),
    numSpeakers: tts.numSpeakers,
  });
} catch (e) {
  self.postMessage({
    type: "error",
    message: "TTS Initialization failed: " + e.message,
  });
}

function getErrorMessage(err) {
  if (err instanceof Error) {
    if (err.stack) {
      return `${err.message}\n${err.stack}`;
    }
    return err.message;
  }

  return `${err}`;
}

self.onmessage = async (e) => {
  const { type, text, sid, speed, genConfig } = e.data;
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
        message: "Generation failed: " + getErrorMessage(err),
      });
    }
  } else if (type === "generateWithConfig") {
    if (!tts) {
      return;
    }
    try {
      const config = Object.assign({}, genConfig || {});
      config.callback = (samples, n, progress) => {
        self.postMessage({
          type: "sherpa-onnx-tts-generation-progress",
          progress: progress,
        });
        return 1;
      };

      const audio = tts.generateWithConfig(text, config);
      const samples = audio.samples;
      const sampleRate = audio.sampleRate;
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
        message: "Generation failed: " + getErrorMessage(err),
      });
    }
  }
};
