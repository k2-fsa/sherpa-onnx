#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint sherpa_onnx_ios.podspec` to validate before publishing.
#
# See also
# https://github.com/google/webcrypto.dart/blob/2010361a106d7a872d90e3dfebfed250e2ede609/ios/webcrypto.podspec#L23-L28
# https://groups.google.com/g/dart-ffi/c/nUATMBy7r0c
Pod::Spec.new do |s|
  s.name             = 'sherpa_onnx_ios'
  s.version          = '1.12.15'
  s.summary          = 'A new Flutter FFI plugin project.'
  s.description      = <<-DESC
A new Flutter FFI plugin project.
                       DESC
  s.homepage         = 'https://github.com/k2-fsa/sherpa-onnx'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Fangjun Kuang' => 'csukuangfj@gmail.com' }

  # This will ensure the source files in Classes/ are included in the native
  # builds of apps using this FFI plugin. Podspec does not support relative
  # paths, so Classes contains a forwarder C file that relatively imports
  # `../src/*` so that the C sources can be shared among all target platforms.
  s.source           = { :path => '.' }
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'
  s.preserve_paths = 'sherpa_onnx.xcframework/**/*'
  s.vendored_frameworks = 'sherpa_onnx.xcframework'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
    }
  s.swift_version = '5.0'
end
