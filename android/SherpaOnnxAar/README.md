# Usage of this project

```
git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.15/sherpa-onnx-v1.12.15-android.tar.bz2
tar xvf sherpa-onnx-v1.12.15-android.tar.bz2

cp -v jniLibs/arm64-v8a/* android/SherpaOnnxAar/sherpa_onnx/src/main/jniLibs/arm64-v8a/
cp -v jniLibs/armeabi-v7a/* android/SherpaOnnxAar/sherpa_onnx/src/main/jniLibs/armeabi-v7a/
cp -v jniLibs/x86/* android/SherpaOnnxAar/sherpa_onnx/src/main/jniLibs/x86/
cp -v jniLibs/x86_64/* android/SherpaOnnxAar/sherpa_onnx/src/main/jniLibs/x86_64/

cd android/SherpaOnnxAar

./gradlew :sherpa_onnx:assembleRelease
ls -lh ./sherpa_onnx/build/outputs/aar/sherpa_onnx-release.aar
cp ./sherpa_onnx/build/outputs/aar/sherpa_onnx-release.aar ../../sherpa-onnx-1.12.15.aar
```
