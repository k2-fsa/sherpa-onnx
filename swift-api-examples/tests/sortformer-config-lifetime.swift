import Foundation

private func withAutoreleasePool<Result>(_ body: () -> Result) -> Result {
  #if canImport(ObjectiveC)
  return autoreleasepool(invoking: body)
  #else
  return body()
  #endif
}

@main
struct SortformerConfigLifetimeTest {
  static func main() throws {
    for path in ["", "m.onnx", String(repeating: "模型-é/", count: 100) + "model.onnx"] {
      let copied = withAutoreleasePool {
        withSherpaOnnxOfflineSpeakerSegmentationSortformerModelConfig(
          model: String(path.reversed().reversed()), threshold: 0.7
        ) { sortformer in
          // Drain a nested pool before C consumes the path, and exercise
          // allocation while the config borrows its UTF-8 storage.
          withAutoreleasePool {
            for _ in 0..<100 {
              _ = (UUID().uuidString as NSString).length
            }
          }
          precondition(sortformer.threshold == 0.7)
          return String(cString: sortformer.model)
        }
      }
      precondition(copied == path)
    }

    enum ExpectedError: Error { case test }
    do {
      try withSherpaOnnxOfflineSpeakerSegmentationSortformerModelConfig(model: "model.onnx") {
        _ in throw ExpectedError.test
      }
      preconditionFailure("The helper must propagate errors")
    } catch ExpectedError.test {}

    // Supply an exported model to additionally check native creation and use
    // after the closure and its autorelease pool have ended.
    if CommandLine.arguments.count == 2 {
      let sd = withAutoreleasePool {
        withSherpaOnnxOfflineSpeakerSegmentationSortformerModelConfig(
          model: String(CommandLine.arguments[1].reversed().reversed())
        ) { sortformer in
          var config = sherpaOnnxOfflineSpeakerDiarizationConfig(
            segmentation: sherpaOnnxOfflineSpeakerSegmentationModelConfig(sortformer: sortformer))
          return SherpaOnnxOfflineSpeakerDiarizationWrapper(config: &config)
        }
      }
      precondition(sd.impl != nil)
      precondition(sd.sampleRate == 16000)
      _ = sd.process(samples: [Float](repeating: 0, count: 1280))
    }
    print("Sortformer config lifetime tests passed")
  }
}
