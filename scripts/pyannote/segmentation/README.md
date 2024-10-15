# File description

Please download test wave files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models

## 0-four-speakers-zh.wav

It is recorded by @csukuangfj

## 1-two-speakers-en.wav

This file is from
https://github.com/pengzhendong/pyannote-onnx/blob/master/data/test_16k.wav
and it contains speeches from two speakers.

Note that we have renamed it from `test_16k.wav` to `1-two-speakers-en.wav`


## 2-two-speakers-en.wav
This file is from
https://huggingface.co/spaces/Xenova/whisper-speaker-diarization

Note that the original file is `./fcf059e3-689f-47ec-a000-bdace87f0113.mp4`.
We use the following commands to convert it to `2-two-speakers-en.wav`.

```bash
ffmpeg -i ./fcf059e3-689f-47ec-a000-bdace87f0113.mp4 -ac 1 -ar 16000 ./2-two-speakers-en.wav
```

## 3-two-speakers-en.wav

This file is from
https://aws.amazon.com/blogs/machine-learning/deploy-a-hugging-face-pyannote-speaker-diarization-model-on-amazon-sagemaker-as-an-asynchronous-endpoint/

Note that the original file is `ML16091-Audio.mp3`. We use the following
commands to convert it to `3-two-speakers-en.wav`


```bash
sox ML16091-Audio.mp3 -r 16k 3-two-speakers-en.wav
```
