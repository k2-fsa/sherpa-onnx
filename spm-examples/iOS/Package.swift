// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "SherpaOnnxExample",
  platforms: [.iOS(.v15)],
  dependencies: [
    .package(url: "https://github.com/k2-fsa/sherpa-onnx", exact: "1.13.7"),
  ],
  targets: [
    .executableTarget(
      name: "SherpaOnnxExample",
      dependencies: [
        .product(name: "sherpa-onnx", package: "sherpa-onnx"),
      ],
      linkerSettings: [
        .linkedLibrary("c++"),
      ]
    )
  ]
)
