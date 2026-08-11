// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "SherpaOnnxExample",
  platforms: [.macOS(.v10_15)],
  dependencies: [
    .package(url: "https://github.com/k2-fsa/sherpa-onnx", exact: "v1.13.5"),
  ],
  targets: [
    .executableTarget(
      name: "SherpaOnnxExample",
      dependencies: [
        .product(name: "sherpa-onnx", package: "sherpa-onnx"),
      ]
    )
  ]
)
