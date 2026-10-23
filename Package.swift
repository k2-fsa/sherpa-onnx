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
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.0-macos-static.xcframework.zip",
      checksum: "5a8b8924c9f4e79a2f6655eef22027b31e3c4e2c57b61aa4862fd1ff5eed9abc"
    ),
    .binaryTarget(
      name: "SherpaOnnxIOS",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.0-ios-static.xcframework.zip",
      checksum: "95550cc9e2d9699eccbc41d6d7462fb83cdfc12dd070d937533f8d0383a38286"
    ),

    // --- Shared binary targets ---
    .binaryTarget(
      name: "SherpaOnnxMacOSShared",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.0-macos-shared.xcframework.zip",
      checksum: "8ef57e8935f3fdb4da52a11f0b5dacee2aa4d525ae145a859c49ae12dee08145"
    ),
    .binaryTarget(
      name: "SherpaOnnxIOSShared",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v2.0.0-ios-shared.xcframework.zip",
      checksum: "8388076ec54af05aa06b4c8696f941d643cbbf48da95d3fc7750b1184315f1b1"
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
        "qwen3-asr.swift",
        "source-separation-spleeter.swift",
        "source-separation-uvr.swift",
        "speaker-diarization.swift",
        "speech-enhancement-dpdfnet.swift",
        "speech-enhancement-gtcrn.swift",
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
        "run-qwen3-asr.sh",
        "run-source-separation-spleeter.sh",
        "run-source-separation-uvr.sh",
        "run-speaker-diarization.sh",
        "run-speech-enhancement-dpdfnet.sh",
        "run-speech-enhancement-gtcrn.sh",
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
