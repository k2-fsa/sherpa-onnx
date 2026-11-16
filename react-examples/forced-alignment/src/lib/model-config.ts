// Model configuration — the wasm assets/ directory holds model directories
// (see wasm/forced-alignment/assets/README.md); one app ships one model, the
// index selects which one (same order as the Android app's ModelConfig.kt).
// The CI sets SELECTED_MODEL_INDEX per published app.

export const SELECTED_MODEL_INDEX = 0;

export interface FaModelInfo {
  /** Model directory name inside the wasm FS, e.g. 'zh-aishell3-pinyin' */
  dir: string;
  /** Display name */
  name: string;
  /** Download URL of the model package */
  url: string;
  /** File name of the package's test wav */
  exampleWav: string;
  /** Transcript of the test wav */
  exampleText: string;
}

const docUrl =
  "https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/mfa/android.md";
export { docUrl };

const faModelsUrl =
  "https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models";

// All Chinese packages ship the same zh.wav, all English ones the same
// en.wav; the transcripts are the ones in the packages' README.md.
// The transcript of the test wav zh.wav (data, not UI).
const zhExampleText = "不要问你的国家能为你做什么，而要问你能为你的国家做什么。";
const enExampleText =
  "ask not what your country can do for you, ask what you can do for your country.";

export const modelInfo: FaModelInfo[] = [
  {
    dir: "zh-aishell3-pinyin",
    name: "Chinese MFA - aishell3 pinyin",
    url: `${faModelsUrl}/zh-aishell3-pinyin.tar.bz2`,
    exampleWav: "zh.wav",
    exampleText: zhExampleText,
  },
  {
    dir: "zh-biaobei-pinyin",
    name: "Chinese MFA - biaobei pinyin",
    url: `${faModelsUrl}/zh-biaobei-pinyin.tar.bz2`,
    exampleWav: "zh.wav",
    exampleText: zhExampleText,
  },
  {
    dir: "zh-large-pinyin",
    name: "Chinese MFA - large pinyin",
    url: `${faModelsUrl}/zh-large-pinyin.tar.bz2`,
    exampleWav: "zh.wav",
    exampleText: zhExampleText,
  },
  {
    dir: "english-mfa-v2.2.1",
    name: "English MFA v2.2.1",
    url: `${faModelsUrl}/english-mfa-v2.2.1.tar.bz2`,
    exampleWav: "en.wav",
    exampleText: enExampleText,
  },
  {
    dir: "english-mfa-v3.0.0",
    name: "English MFA v3.0.0",
    url: `${faModelsUrl}/english-mfa-v3.0.0.tar.bz2`,
    exampleWav: "en.wav",
    exampleText: enExampleText,
  },
  {
    dir: "english-mfa-v3.1.0",
    name: "English MFA v3.1.0",
    url: `${faModelsUrl}/english-mfa-v3.1.0.tar.bz2`,
    exampleWav: "en.wav",
    exampleText: enExampleText,
  },
];

export function currentModel(index: number = SELECTED_MODEL_INDEX): FaModelInfo {
  return modelInfo[Math.min(index, modelInfo.length - 1)];
}
