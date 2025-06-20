# Introduction

Please refer to
https://k2-fsa.github.io/sherpa/onnx/lazarus/generate-subtitles.html
for how to build the project in this directory.

You can find pre-built APPs from this directory at
https://k2-fsa.github.io/sherpa/onnx/lazarus/pre-built-app.html



## notes for developers

By default, it uses static libs for Linux and macOS. To change that,
open Lazarus IDE, select `Project`, `Project options`, `Compiler options`,
change the `Build modes` to `Release-Linux`.

The `Release-Linux` mode is defined in the file `generate_subtitles.lpi`.
It defines a macro `SHERPA_ONNX_USE_SHARED_LIBS`.
