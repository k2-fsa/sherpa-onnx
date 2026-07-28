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
      checksum: "TO_BE_UPDATED"
    ),
    .binaryTarget(
      name: "sherpa-onnx-ios",
      url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.4/sherpa-onnx-v1.13.4-ios.xcframework.zip",
      checksum: "TO_BE_UPDATED"
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
