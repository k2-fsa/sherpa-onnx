# Real-time speech recognition

This APP supports the following platforms:

  - Windows
  - macOS
  - Android
  - iOS

Note that it does not support Linux since we are using
the package [record](https://pub.dev/packages/record), which does not
support streaming recording on Linux.

If you can find a recording package
that works on Linux, please let us know and we will update this app to support Linux.

## Getting Started

Remember to use the following steps to download a model. Otherwise, you would
get errors after you start and run the app.

###  1. Select a streaming model

Please visit <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>
to download a streaming ASR model.

You can find introductions about each streaming model at
<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html>


Note: `Streaming` is the same as `Online` in this context.

### 2. Let the code know which model you are using

We have pre-configured some streaming models in the following file

<https://github.com/k2-fsa/sherpa-onnx/blob/master/flutter-examples/streaming_asr/lib/online_model.dart>

If you select a model that is not in the above file, please add it to the above file
by yourself by following how existing models are added.

Then you need to update

<https://github.com/k2-fsa/sherpa-onnx/blob/master/flutter-examples/streaming_asr/lib/streaming_asr.dart#L16>

```
final type = 0;
```

Please change ``type`` accordingly.

You also need to change [./pubspec.yaml](./pubspec.yaml) so that your APP knows where to find it.
Please see the example below for how to do that.

### 3. Place your downloaded model inside the directory assets

The downloaded model has to be placed in the [assets](./assets) directory.

**HINT**: Please delete files that are not needed by the code. Otherwise, you put
unnecessary files in your APP and it will significantly increase the size of your APP.

## Example

Suppose you have selected the following model

<https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2>

Please use the following steps to make it available in your APP.

 - 1. Change [online_model.dart](./lib/online_model.dart)

    This model is already in the file and its type is `0`, so there is no need to change this file.

 - 2. Change [streaming_asr.dart](./lib/streaming_asr.dart)

    The default value for `type` is 0 and our model has also a type of `0`, so there is no need to change this file.

 - 3. Change [pubspec.yaml](./pubspec.yaml)

   At the end of [pubspec.yaml](./pubspec.yaml), please change it exactly like below:

```
  assets:
    - assets/
    - assets/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/
```

  - 4. Download the model to the [./assets](./assets) directory.

```
cd assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

# Remeber to remove unused files.
rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/README.md
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/bpe*
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.int8.onnx
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx
```

Your [assets](./assets) directory should look like below at the end.

```
assets/
└── sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
    ├── decoder-epoch-99-avg-1.onnx
    ├── encoder-epoch-99-avg-1.int8.onnx
    ├── joiner-epoch-99-avg-1.onnx
    └── tokens.txt

1 directory, 4 files
```

  - 5. Run it!

    For instance

      - `flutter run -d macos` for macOS.

      - `flutter run -d windows` for windows.
