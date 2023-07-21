# Introduction

This examples shows how to use the golang package of [sherpa-onnx][sherpa-onnx]
for real-time speech recognition from microphone.

It uses <https://github.com/gordonklaus/portaudio>
to read the microphone and you have to install `portaudio` first.

On macOS, you can use

```
brew install portaudio
```

and it will install `portaudio` into `/usr/local/Cellar/portaudio/19.7.0`.
You need to set the following environment variable
```
export PKG_CONFIG_PATH=/usr/local/Cellar/portaudio/19.7.0
```

so that `pkg-config --cflags --libs portaudio-2.0` can run successfully.

[sherpa-onnx]: https://github.com/k2-fsa/sherpa-onnx
