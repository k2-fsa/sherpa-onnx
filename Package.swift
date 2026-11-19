// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "sherpa-onnx",
  platforms: [
    .iOS(.v15),
    .macOS(.v10_15),
  ],
  products: [
    // Static xcframework (default)
    .library(name: "sherpa-onnx", targets: ["SherpaOnnx"]),
    // Shared/dynamic xcframework
    .library(name: "sherpa-onnx-shared", targets: ["SherpaOnnxShared"]),
  ],
  dependencies: [
    .package(url: "https://github.com/csukuangfj/onnxruntime-libs", exact: "1.28.2")
  ],
  targets: [
    // --- Static binary targets ---
    .binaryTarget(
      name: "SherpaOnnxMacOS",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.2-macos-static.xcframework.zip",
      checksum: "d538fd588280db60a514e40729a2526d4726ca3cedcd623cac0d25026d4495d6"
    ),
    .binaryTarget(
      name: "SherpaOnnxIOS",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.2-ios-static.xcframework.zip",
      checksum: "2f02c8255ae152da859753a7ec9570d140bbab01c38f3f8829531278f57b3d60"
    ),

    // --- Shared binary targets ---
    .binaryTarget(
      name: "SherpaOnnxMacOSShared",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.2-macos-shared.xcframework.zip",
      checksum: "68eae54e50ee05f3978b677ee0ab3fd2d138890ca1ef6b81ed520943e09f0359"
    ),
    .binaryTarget(
      name: "SherpaOnnxIOSShared",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.2-ios-shared.xcframework.zip",
      checksum: "682fa7832ca4f063c3b1136fcdde153a37ded8c6c10553bae47b3a45fe4bdbec"
    ),

    // --- Static wrapper target (default) ---
    .target(
      name: "SherpaOnnx",
      dependencies: [
        .product(
          name: "onnxruntime-macos", package: "onnxruntime-libs",
          condition: .when(platforms: [.macOS])),
        .product(
          name: "onnxruntime-ios", package: "onnxruntime-libs", condition: .when(platforms: [.iOS])),
        .target(name: "SherpaOnnxMacOS", condition: .when(platforms: [.macOS])),
        .target(name: "SherpaOnnxIOS", condition: .when(platforms: [.iOS])),
      ],
      path: "swift-api-examples",
      exclude: [
        "SherpaOnnx-Bridging-Header.h",
        "add-punctuation-online.swift",
        "add-punctuations.swift",
        "cohere-transcribe.swift",
        "compute-speaker-embeddings.swift",
        "decode-file-non-streaming.swift",
        "decode-file-sense-voice-with-hr.swift",
        "decode-file-t-one-streaming.swift",
        "decode-file.swift",
        "dolphin-ctc-asr.swift",
        "fire-red-asr-ctc.swift",
        "fire-red-asr.swift",
        "funasr-nano.swift",
        "generate-subtitles.swift",
        "keyword-spotting-from-file.swift",
        "medasr-ctc.swift",
        "moonshine-v2-asr.swift",
        "omnilingual-asr-ctc.swift",
        "online-speech-enhancement-dpdfnet.swift",
        "online-speech-enhancement-gtcrn.swift",
        "online-speech-enhancement-voxlock-with-voice-sample.swift",
        "online-speech-enhancement-voxlock.swift",
        "qwen3-asr.swift",
        "source-separation-spleeter.swift",
        "source-separation-uvr.swift",
        "speaker-diarization.swift",
        "speech-enhancement-dpdfnet.swift",
        "speech-enhancement-gtcrn.swift",
        "speech-enhancement-voxlock-with-voice-sample.swift",
        "speech-enhancement-voxlock.swift",
        "spoken-language-identification.swift",
        "streaming-hlg-decode-file.swift",
        "test-version.swift",
        "tts-kitten-en-lexicon.swift",
        "tts-kokoro-en-lexicon.swift",
        "tts-kokoro-zh-en-lexicon.swift",
        "tts-matcha-en-lexicon.swift",
        "tts-matcha-zh-lexicon.swift",
        "tts-pocket-en.swift",
        "tts-supertonic-en.swift",
        "tts-vits-lexicon.swift",
        "tts-zipvoice-lexicon.swift",
        "wenet-ctc-asr.swift",
        "zipformer-ctc-asr.swift",
        "zipformer-ctc-hlg.swift",
        "run-add-punctuations-online.sh",
        "run-add-punctuations.sh",
        "run-cohere-transcribe.sh",
        "run-compute-speaker-embeddings.sh",
        "run-decode-file-non-streaming.sh",
        "run-decode-file-sense-voice-with-hr.sh",
        "run-decode-file-t-one-streaming.sh",
        "run-decode-file.sh",
        "run-dolphin-ctc-asr.sh",
        "run-fire-red-asr-ctc.sh",
        "run-fire-red-asr.sh",
        "run-funasr-nano-asr.sh",
        "run-generate-subtitles-ten-vad.sh",
        "run-generate-subtitles.sh",
        "run-keyword-spotting-from-file.sh",
        "run-medasr-ctc-asr.sh",
        "run-moonshine-v2-asr.sh",
        "run-omnilingual-asr-ctc-asr.sh",
        "run-online-speech-enhancement-dpdfnet.sh",
        "run-online-speech-enhancement-gtcrn.sh",
        "run-online-speech-enhancement-voxlock-with-voice-sample.sh",
        "run-online-speech-enhancement-voxlock.sh",
        "run-qwen3-asr.sh",
        "run-source-separation-spleeter.sh",
        "run-source-separation-uvr.sh",
        "run-speaker-diarization.sh",
        "run-speech-enhancement-dpdfnet.sh",
        "run-speech-enhancement-gtcrn.sh",
        "run-speech-enhancement-voxlock-with-voice-sample.sh",
        "run-speech-enhancement-voxlock.sh",
        "run-spoken-language-identification.sh",
        "run-streaming-hlg-decode-file.sh",
        "run-test-version.sh",
        "run-tts-kitten-en-lexicon.sh",
        "run-tts-kokoro-en-lexicon.sh",
        "run-tts-kokoro-zh-en-lexicon.sh",
        "run-tts-matcha-en-lexicon.sh",
        "run-tts-matcha-zh-lexicon.sh",
        "run-tts-pocket-en.sh",
        "run-tts-supertonic-en.sh",
        "run-tts-vits-lexicon.sh",
        "run-tts-zipvoice-lexicon.sh",
        "run-wenet-ctc-asr.sh",
        "run-zipformer-ctc-asr.sh",
        "run-zipformer-ctc-hlg.sh",
      ],
      sources: ["SherpaOnnx.swift"],
      linkerSettings: [
        .linkedLibrary("c++")
      ]
    ),

    // --- Shared wrapper target ---
    .target(
      name: "SherpaOnnxShared",
      dependencies: [
        .product(
          name: "onnxruntime-macos-shared", package: "onnxruntime-libs",
          condition: .when(platforms: [.macOS])),
        .product(
          name: "onnxruntime-ios-shared", package: "onnxruntime-libs",
          condition: .when(platforms: [.iOS])),
        .target(name: "SherpaOnnxMacOSShared", condition: .when(platforms: [.macOS])),
        .target(name: "SherpaOnnxIOSShared", condition: .when(platforms: [.iOS])),
      ],
      path: "Sources/SherpaOnnxShared",
      sources: ["SherpaOnnx.swift"],
      linkerSettings: [
        .linkedLibrary("c++")
      ]
    ),
  ]
)
