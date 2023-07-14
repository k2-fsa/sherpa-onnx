# Speech recognition with Visual C++ MFC

This directory contains examples showing how to use Next-gen Kaldi in MFC
for speech recognition.

Caution: You need to use Windows and install Visual Studio 2022 in order to
compile it.

Hint: If you don't want to install Visual Studio, you can find below
about how to download pre-compiled `exe`.

We use bash script below to demonstrate how to use it. Please change
the commands accordingly for Windows.

## How to compile


First, we need to compile sherpa-onnx:

```bash
mkdir -p $HOME/open-source
cd $HOME/open-source

git clone https://github.com/k2-fsa/sherpa-onnx
cd sherpa-onnx
mkdir build

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=./install ..
cmake --build . --config Release --target install
cd ../mfc-examples

msbuild ./mfc-examples.sln /property:Configuration=Release /property:Platform=x64

# now run the program

./x64/Release/StreamingSpeechRecognition.exe
./x64/Release/NonStreamingSpeechRecognition.exe
```

If you don't want to compile the project by yourself, you can download
pre-compiled `exe` from https://github.com/k2-fsa/sherpa-onnx/releases

For instance, you can use the following addresses:

  - https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.5.1/sherpa-onnx-streaming-v1.5.1.exe
  - https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.5.1/sherpa-onnx-non-streaming-v1.5.1.exe
