// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "sherpa_onnx_ios",
  platforms: [
    .iOS(.v13)
  ],
  products: [
    .library(name: "sherpa-onnx-ios", targets: ["sherpa_onnx_ios"])
  ],
  dependencies: [
    .package(name: "FlutterFramework", path: "../FlutterFramework")
  ],
  targets: [
    .binaryTarget(
      name: "SherpaOnnxC",
      path: "SherpaOnnxC.xcframework"
    ),
    .target(
      name: "sherpa_onnx_ios",
      dependencies: [
        .product(name: "FlutterFramework", package: "FlutterFramework"),
        "SherpaOnnxC",
      ]
    )
  ]
)
