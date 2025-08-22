# Introduction

This folder is for developers only.

## sherpa-onnx-core

It contains the scripts for building the package sherpa-onnx-core.

```
python3 setup.py bdist_wheel --plat-name=macosx_10_15_x86_64
python3 setup.py bdist_wheel --plat-name=macosx_11_0_arm64
python3 setup.py bdist_wheel --plat-name=macosx_11_0_universal2
python3 setup.py bdist_wheel --plat-name=macosx_10_15_universal2

python3 setup.py bdist_wheel --plat-name=win_amd64
python3 setup.py bdist_wheel --plat-name=win32

python3 setup.py bdist_wheel --plat-name=manylinux2014_x86_64
python3 setup.py bdist_wheel --plat-name=manylinux2014_aarch64
python3 setup.py bdist_wheel --plat-name=linux_armv7l
```

## sherpa-onnx-bin
