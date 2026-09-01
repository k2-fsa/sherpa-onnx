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
    .package(url: "https://github.com/csukuangfj/onnxruntime-libs", exact: "1.28.1")
  ],
  targets: [
    // --- Static binary targets ---
    .binaryTarget(
      name: "SherpaOnnxMacOS",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v1.13.7-macos-static.xcframework.zip",
      checksum: "f22753dd36e37c59fbae7667b16bba75b4c8c74e953e793f080a005a56e2ce0b"
    ),
    .binaryTarget(
      name: "SherpaOnnxIOS",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v1.13.7-ios-static.xcframework.zip",
      checksum: "a808329c49da521b3af707da2e1a9d5b0a4595b2549ffdc771f2f560f012fd3d"
    ),

    // --- Shared binary targets ---
    .binaryTarget(
      name: "SherpaOnnxMacOSShared",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v1.13.7-macos-shared.xcframework.zip",
      checksum: "bf6c1b4952a744e2df947fbd9beabb7ea050f9f537b5eb9d7ffeb35c2967de25"
    ),
    .binaryTarget(
      name: "SherpaOnnxIOSShared",
      url:
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v1.13.7-ios-shared.xcframework.zip",
      checksum: "f1608557c576620287da091d549a3b61a64409412ee4cf82c3e75dbf9e52023b"
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
        "tts-kitten-en.swift",
        "tts-kokoro-en.swift",
        "tts-kokoro-zh-en.swift",
        "tts-matcha-en.swift",
        "tts-matcha-zh.swift",
        "tts-pocket-en.swift",
        "tts-supertonic-en.swift",
        "tts-vits.swift",
        "tts-zipvoice.swift",
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
        "run-tts-kitten-en.sh",
        "run-tts-kokoro-en.sh",
        "run-tts-kokoro-zh-en.sh",
        "run-tts-matcha-en.sh",
        "run-tts-matcha-zh.sh",
        "run-tts-pocket-en.sh",
        "run-tts-supertonic-en.sh",
        "run-tts-vits.sh",
        "run-tts-zipvoice.sh",
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
