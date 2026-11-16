package com.k2fsa.sherpa.onnx.fa

// Which model this APK bundles (0-5). scripts/apk/build-apk-forced-alignment.sh
// sets this per APK with sed; change it by hand when building locally.
const val SELECTED_MODEL_INDEX = 0

/**
 * Info about one forced-alignment model package.
 *
 * The model directory [dir] must exist in app/src/main/assets/ (extract a
 * [url] tarball there; it holds meta.json, final.alimdl, tree, lda.mat,
 * phones.txt, lexicon.txt and the test wav).
 */
data class FaModelInfo(
    val name: String,
    val dir: String,
    val url: String,
    val doc: String,
    val exampleWav: String,
    val exampleText: String,
)

private const val faModelsUrl =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models"

private const val docUrl =
    "https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/mfa/android.md"

// All Chinese packages ship the same zh.wav, all English ones the same en.wav;
// the transcripts below are the ones in the packages' README.md.
private const val zhExampleText = "不要问你的国家能为你做什么，而要问你能为你的国家做什么。"

private const val enExampleText =
    "ask not what your country can do for you, ask what you can do for your country."

val modelInfo = listOf(
    FaModelInfo(
        name = "Chinese MFA - aishell3 pinyin",
        dir = "zh-aishell3-pinyin",
        url = "$faModelsUrl/zh-aishell3-pinyin.tar.bz2",
        doc = docUrl,
        exampleWav = "zh.wav",
        exampleText = zhExampleText,
    ),
    FaModelInfo(
        name = "Chinese MFA - biaobei pinyin",
        dir = "zh-biaobei-pinyin",
        url = "$faModelsUrl/zh-biaobei-pinyin.tar.bz2",
        doc = docUrl,
        exampleWav = "zh.wav",
        exampleText = zhExampleText,
    ),
    FaModelInfo(
        name = "Chinese MFA - large pinyin",
        dir = "zh-large-pinyin",
        url = "$faModelsUrl/zh-large-pinyin.tar.bz2",
        doc = docUrl,
        exampleWav = "zh.wav",
        exampleText = zhExampleText,
    ),
    FaModelInfo(
        name = "English MFA v2.2.1",
        dir = "english-mfa-v2.2.1",
        url = "$faModelsUrl/english-mfa-v2.2.1.tar.bz2",
        doc = docUrl,
        exampleWav = "en.wav",
        exampleText = enExampleText,
    ),
    FaModelInfo(
        name = "English MFA v3.0.0",
        dir = "english-mfa-v3.0.0",
        url = "$faModelsUrl/english-mfa-v3.0.0.tar.bz2",
        doc = docUrl,
        exampleWav = "en.wav",
        exampleText = enExampleText,
    ),
    FaModelInfo(
        name = "English MFA v3.1.0",
        dir = "english-mfa-v3.1.0",
        url = "$faModelsUrl/english-mfa-v3.1.0.tar.bz2",
        doc = docUrl,
        exampleWav = "en.wav",
        exampleText = enExampleText,
    ),
)

val currentModel: FaModelInfo
    get() = modelInfo[SELECTED_MODEL_INDEX]
