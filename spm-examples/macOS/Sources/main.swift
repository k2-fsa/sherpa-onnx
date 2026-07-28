import SherpaOnnxC

let version = String(cString: SherpaOnnxGetVersionStr())
let gitSha1 = String(cString: SherpaOnnxGetGitSha1())
let gitDate = String(cString: SherpaOnnxGetGitDate())
let ortVersion = String(cString: SherpaOnnxGetOnnxruntimeVersionStr())

print("sherpa-onnx version : \(version)")
print("sherpa-onnx Git SHA1: \(gitSha1)")
print("sherpa-onnx Git date: \(gitDate)")
print("onnxruntime version : \(ortVersion)")
