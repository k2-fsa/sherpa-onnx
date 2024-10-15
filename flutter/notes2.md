# Some use commands while learning flutter/dart

## macOS

1. Build required libraries

```bash
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX=./install -DBUILD_SHARED_LIBS=ON -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" ..
make install
cd ../sherpa-onnx/flutter/
cp -v  ../../build/install/lib/lib* ./macos/
```

2. Test for speaker identification

```bash
cd sherpa-onnx/sherpa-onnx/flutter/example
mkdir assets
```


## Useful commands
```
flutter pub publish --dry-run
flutter run -d macos
flutter run -d linux
flutter run -d windows

flutter build macos

flutter run --release -d macos

# add platform to an existing project
flutter create --platforms=windows,macos,linux .

dart analyze

FLUTTER_XCODE_ARCHS=arm64
FLUTTER_XCODE_ARCHS=x86_64
```

## Examples

  - https://dart.dev/tools/pub/automated-publishing

     Use GitHub actions to publish

  - https://dart.dev/tools/pub/pubspec

     It describes the format of ./pubspec.yaml

  - https://github.com/folksable/blurhash_ffi/

      It supports ios, android, linux, macos, and windows.

 - https://github.com/alexmercerind/dart_vlc
 - https://github.com/dart-lang/native/tree/main/pkgs/jni
