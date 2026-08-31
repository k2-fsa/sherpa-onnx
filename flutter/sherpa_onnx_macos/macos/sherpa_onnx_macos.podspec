#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint sherpa_onnx_macos.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'sherpa_onnx_macos'
  s.version          = '1.13.7'
  s.summary          = 'sherpa-onnx Flutter FFI plugin project.'
  s.description      = <<-DESC
sherpa-onnx Flutter FFI plugin project.
                       DESC
  s.homepage         = 'https://github.com/k2-fsa/sherpa-onnx'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Fangjun Kuang' => 'csukuangfj@gmail.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.dependency 'FlutterMacOS'
  # The dylib lives inside the xcframework (single copy, no duplication).
  # CocoaPods embeds it; SPM resolves it from the same xcframework.
  s.vendored_libraries = 'sherpa_onnx_macos/sherpa-onnx.xcframework/macos-arm64_x86_64/libsherpa-onnx-c-api.dylib'

  s.platform = :osx, '10.15'
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES' }
  s.swift_version = '5.0'
end
