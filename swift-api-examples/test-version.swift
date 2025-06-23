func run() {
  let version = getSherpaOnnxVersion()
  let gitSha1 = getSherpaOnnxGitSha1()
  let gitDate = getSherpaOnnxGitDate()
  print("sherpa-onnx version: \(version)")
  print("sherpa-onnx gitSha1: \(gitSha1)")
  print("sherpa-onnx gitDate: \(gitDate)")
}

@main
struct App {
  static func main() {
    run()
  }
}
