// sherpa-onnx wasm engine loading for forced alignment.
//
// Loading/initialization phases:
//   0. inject the API wrapper script (classic script; classes mount on window)
//   1. stream-fetch the inference core .wasm with progress -> Module.wasmBinary
//   2. stream-fetch the model package .data with progress
//      -> Module.getPreloadedPackage (the model files are packed in it)
//   3. set window.Module hooks and inject the emscripten main script
//   4. create the ForcedAligner once ("ready")
//
// loadSherpaEngineOnce() is a singleton with safe retry before the main
// script is injected.

import { currentModel } from "../model-config";

const SHERPA_BASE = "/sherpa";

const MAIN_JS = "sherpa-onnx-wasm-main-fa.js";
const FA_JS = "sherpa-onnx-fa.js";
const MAIN_WASM = "sherpa-onnx-wasm-main-fa.wasm";
const MAIN_DATA = "sherpa-onnx-wasm-main-fa.data";

export type LoadStage = "wasm" | "data" | "init" | "ready";

export interface LoadProgress {
  stage: LoadStage;
  loaded: number;
  total: number;
  message: string;
}

export interface WordInterval {
  start: number;
  end: number;
  text: string;
}

export interface AlignResult {
  duration: number;
  textgrid: string;
  words: WordInterval[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyModule = any;

export interface SherpaEngine {
  module: AnyModule;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  aligner: any;
  modelDirs: string[];
}

export interface LoadOptions {
  onProgress?: (p: LoadProgress) => void;
  /** Index into model-config.ts's modelInfo; the directory must exist. */
  modelIndex?: number;
}

const MB = 1024 * 1024;
const MB_FMT = (n: number) => `${(n / MB).toFixed(1)}MB`;

async function fetchWithProgress(
  url: string,
  onLoaded: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download the resource: HTTP ${response.status}`);
  }
  const total = Number(response.headers.get("content-length") || 0);
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onLoaded(buffer.byteLength, buffer.byteLength);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.length;
      onLoaded(loaded, total);
    }
  }
  const merged = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged.buffer;
}

/** Inject a classic <script> and wait for onload (idempotent per src). */
function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      `script[src="${src}"]`,
    );
    if (existing && existing.dataset.loaded === "1") {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = false;
    script.onload = () => {
      script.dataset.loaded = "1";
      resolve();
    };
    script.onerror = () => {
      if (script.parentNode) script.parentNode.removeChild(script);
      reject(new Error(`Failed to load the script: ${src}`));
    };
    document.head.appendChild(script);
  });
}

let enginePromise: Promise<SherpaEngine> | null = null;

export function loadSherpaEngineOnce(
  options: LoadOptions = {},
): Promise<SherpaEngine> {
  if (!enginePromise) {
    enginePromise = doLoad(options).catch((err) => {
      // safe to retry while the main script has not been injected
      enginePromise = null;
      throw err;
    });
  }
  return enginePromise;
}

async function doLoad(options: LoadOptions): Promise<SherpaEngine> {
  const { onProgress, modelIndex = 0 } = options;
  const report = (
    stage: LoadStage,
    loaded: number,
    total: number,
    message: string,
  ) => onProgress?.({ stage, loaded, total, message });

  // ---- phase 0: the API wrapper (pure function/class definitions) ----
  await loadScript(`${SHERPA_BASE}/${FA_JS}`);

  // ---- phase 1: the inference core .wasm ----
  report("wasm", 0, 0, "Downloading the inference core (wasm)…");
  const wasmBinary = await fetchWithProgress(
    `${SHERPA_BASE}/${MAIN_WASM}`,
    (loaded, total) =>
      report(
        "wasm",
        loaded,
        total,
        `Downloading the inference core ${MB_FMT(loaded)}${total ? " / " + MB_FMT(total) : ""}`,
      ),
  );

  // ---- phase 2: the model package .data (models are packed inside) ----
  report("data", 0, 0, "Downloading the model package (wasm data)…");
  const dataBuffer = await fetchWithProgress(
    `${SHERPA_BASE}/${MAIN_DATA}`,
    (loaded, total) =>
      report(
        "data",
        loaded,
        total,
        `Downloading the model ${MB_FMT(loaded)}${total ? " / " + MB_FMT(total) : ""}`,
      ),
  );

  // ---- phase 3: configure Module and inject the main script ----
  // The emscripten main script reads window.Module synchronously while
  // loading, so both big files must be downloaded before this point.
  const runtimeReady = new Promise<void>((resolve, reject) => {
    window.Module = {
      locateFile: (path: string) => `${SHERPA_BASE}/${path}`,
      wasmBinary,
      getPreloadedPackage: () => dataBuffer,
      setStatus: (status: string) => {
        if (status === "Running...") {
          report("init", 0, 0, "Initializing the inference engine (CPU, single thread)…");
        }
      },
      onRuntimeInitialized: () => resolve(),
      print: (text: string) => console.log("[sherpa]", text),
      printErr: (text: string) => console.warn("[sherpa]", text),
    };
    setTimeout(() => reject(new Error("Engine initialization timed out, please retry")), 300_000);
  });

  report("init", 0, 0, "Starting the inference engine…");
  await loadScript(`${SHERPA_BASE}/${MAIN_JS}`);
  await runtimeReady;

  const module = window.Module as AnyModule;
  if (!module || !module.HEAPF32) {
    throw new Error("The WASM runtime was not initialized correctly");
  }

  // ---- phase 4: create the forced aligner ----
  const ForcedAlignerCtor = window.ForcedAligner;
  if (!ForcedAlignerCtor) {
    throw new Error("ForcedAligner is undefined (sherpa-onnx-fa.js failed to load)");
  }
  const modelDirs = listModelDirs(module);
  const model = currentModel(modelIndex);
  const dir = modelDirs.includes(model.dir)
    ? model.dir
    : modelDirs[Math.min(modelIndex, modelDirs.length - 1)];
  if (!dir) {
    throw new Error("No model in the wasm package; see wasm/forced-alignment/assets/README.md");
  }
  const aligner = new ForcedAlignerCtor(
    { modelDir: dir, lexicon: `${dir}/lexicon.txt` },
    module,
  );

  report("ready", 0, 0, "Engine ready — you can start aligning");
  return { module, aligner, modelDirs };
}

/** Version info baked into the wasm at build time (like the Android app's Info tab). */
export interface VersionInfo {
  version: string;
  gitSha1: string;
  gitDate: string;
  onnxruntime: string;
}

export function readVersionInfo(module: AnyModule): VersionInfo {
  const read = (fn: string): string => {
    const ptr = module[`_${fn}`] ? module[`_${fn}`]() : 0;
    return ptr ? module.UTF8ToString(ptr) : "unknown";
  };
  return {
    version: read("SherpaOnnxGetVersionStr"),
    gitSha1: read("SherpaOnnxGetGitSha1"),
    gitDate: read("SherpaOnnxGetGitDate"),
    onnxruntime: read("SherpaOnnxGetOnnxruntimeVersionStr"),
  };
}

/** Model directories in the wasm FS = directories containing meta.json. */
export function listModelDirs(module: AnyModule): string[] {
  const candidates = [
    "zh-aishell3-pinyin",
    "zh-biaobei-pinyin",
    "zh-large-pinyin",
    "english-mfa-v2.2.1",
    "english-mfa-v3.0.0",
    "english-mfa-v3.1.0",
  ];
  return candidates.filter((d) => {
    try {
      return module.FS.readdir(`/${d}`).indexOf("meta.json") >= 0;
    } catch {
      return false;
    }
  });
}

/** Reads the model package's test wav from the wasm FS. */
export function readExampleWav(module: AnyModule, dir: string, name: string): Uint8Array {
  return module.FS.readFile(`/${dir}/${name}`);
}

/**
 * Runs the forced alignment of one utterance.
 *
 * @param samples float32 mono samples in [-1, 1]
 * @param text the transcript of the whole utterance
 */
export function align(
  engine: SherpaEngine,
  samples: Float32Array,
  sampleRate: number,
  text: string,
): AlignResult {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const stream: any = engine.aligner.createStream();
  try {
    stream.acceptWaveform(sampleRate, samples);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const raw: any = engine.aligner.compute(stream, text);
    const words: WordInterval[] = [];
    for (const tier of raw.tiers || []) {
      if (tier.name === "words") {
        for (const itv of tier.intervals) {
          if (itv.text) words.push(itv);
        }
      }
    }
    return { duration: raw.duration, textgrid: raw.textgrid, words };
  } finally {
    stream.free();
  }
}
