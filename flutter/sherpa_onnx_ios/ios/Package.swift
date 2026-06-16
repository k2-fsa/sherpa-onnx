// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "sherpa_onnx_ios",
    platforms: [.iOS(.v13)],
    products: [
        .library(name: "sherpa_onnx_ios", targets: ["sherpa_onnx_ios"]),
    ],
    targets: [
        .binaryTarget(
            name: "sherpa_onnx_ios",
            path: "sherpa-onnx.xcframework"
        ),
    ]
)
