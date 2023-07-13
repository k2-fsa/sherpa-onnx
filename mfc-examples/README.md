# Speech recognition with Visual C++ MFC

This directory contains examples showing how to use Next-gen Kaldi in MFC
for speech recognition.

Caution: You need to use Windows and install Visual Studio in order to run it.
We use bash script below to demonstrate how to use it. Please change
the commands accordingly for Windows.

## Streaming speech recognition

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
```

Note that we also need to download pre-trained models. Please
refer to https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
for a list of streaming models.

We use the following model for demonstration.

```bash
cd $HOME/open-source/sherpa-onnx/mfc-examples/x64/Release
wget https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615/resolve/main/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx
wget https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615/resolve/main/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx
wget https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615/resolve/main/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx
wget https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615/resolve/main/data/lang_char/tokens.txt

# now rename
mv encoder-epoch-12-avg-4-chunk-16-left-128.onnx encoder.onnx
mv decoder-epoch-12-avg-4-chunk-16-left-128.onnx decoder.onnx
mv joiner-epoch-12-avg-4-chunk-16-left-128.onnx joiner.onnx

# Now run it!
./StreamingSpeechRecognition.exe
```
