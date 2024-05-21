# Usage

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
```
