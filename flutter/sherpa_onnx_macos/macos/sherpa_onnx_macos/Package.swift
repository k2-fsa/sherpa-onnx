// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "sherpa_onnx_macos",
  platforms: [
    .macOS(.v10_15)
  ],
  products: [
    .library(name: "sherpa-onnx-macos", targets: ["sherpa_onnx_macos"])
  ],
  dependencies: [
    .package(name: "FlutterFramework", path: "../FlutterFramework")
  ],
  targets: [
    .binaryTarget(
      name: "SherpaOnnxC",
      path: "sherpa-onnx.xcframework"
    ),
    .target(
      name: "sherpa_onnx_macos",
      dependencies: [
        .product(name: "FlutterFramework", package: "FlutterFramework"),
        "SherpaOnnxC",
      ]
    )
  ]
)
