// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "sherpa-onnx",
  platforms: [
    .iOS(.v13),
    .macOS(.v10_15),
  ],
  products: [
    .library(
      name: "sherpa-onnx",
      targets: ["sherpa-onnx"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/csukuangfj/onnxruntime-libs", branch: "master"),
  ],
  targets: [
    .binaryTarget(
      name: "sherpa-onnx-macos",
      url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-v1.13.4-macos.xcframework.zip",
      checksum: "842a75098671aa6c83d50eb663af40e65d43456fe43b1f4397fe387f3ba0db89"
    ),
    .binaryTarget(
      name: "sherpa-onnx-ios",
      url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-v1.13.4-ios.xcframework.zip",
      checksum: "64a38a7463a1bddf773c7fff058d778754a5b0b227f8706d1e82bbaf5ed97b50"
    ),
    .target(
      name: "sherpa-onnx",
      dependencies: [
        .product(name: "onnxruntime-macos", package: "onnxruntime-libs", condition: .when(platforms: [.macOS])),
        .product(name: "onnxruntime-ios", package: "onnxruntime-libs", condition: .when(platforms: [.iOS])),
        "sherpa-onnx-macos",
        "sherpa-onnx-ios",
      ],
      path: "swift-api-examples",
      exclude: [
        "run-*.sh",
        "SherpaOnnx-Bridging-Header.h",
      ],
      sources: ["SherpaOnnx.swift"],
      linkerSettings: [
        .linkedLibrary("c++"),
      ]
    ),
  ]
)
