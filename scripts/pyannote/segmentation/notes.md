
# config.yaml


```yaml
task:
  _target_: pyannote.audio.tasks.SpeakerDiarization
  duration: 10.0
  max_speakers_per_chunk: 3
  max_speakers_per_frame: 2
model:
  _target_: pyannote.audio.models.segmentation.PyanNet
  sample_rate: 16000
  num_channels: 1
  sincnet:
    stride: 10
  lstm:
    hidden_size: 128
    num_layers: 4
    bidirectional: true
    monolithic: true
  linear:
    hidden_size: 128
    num_layers: 2
```

# Model architecture of ./pytorch_model.bin

`print(model)`:

```python3
PyanNet(
  (sincnet): SincNet(
    (wav_norm1d): InstanceNorm1d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    (conv1d): ModuleList(
      (0): Encoder(
        (filterbank): ParamSincFB()
      )
      (1): Conv1d(80, 60, kernel_size=(5,), stride=(1,))
      (2): Conv1d(60, 60, kernel_size=(5,), stride=(1,))
    )
    (pool1d): ModuleList(
      (0-2): 3 x MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    )
    (norm1d): ModuleList(
      (0): InstanceNorm1d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (1-2): 2 x InstanceNorm1d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
    )
  )
  (lstm): LSTM(60, 128, num_layers=4, batch_first=True, dropout=0.5, bidirectional=True)
  (linear): ModuleList(
    (0): Linear(in_features=256, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=128, bias=True)
  )
  (classifier): Linear(in_features=128, out_features=7, bias=True)
  (activation): LogSoftmax(dim=-1)
)
```

```python3
>>> list(model.specifications)
[Specifications(problem=<Problem.MONO_LABEL_CLASSIFICATION: 1>, resolution=<Resolution.FRAME: 1>, duration=10.0, min_duration=None, warm_up=(0.0, 0.0), classes=['speaker#1', 'speaker#2', 'speaker#3'], powerset_max_classes=2, permutation_invariant=True)]
```

```python3
>>> model.hparams
"linear":       {'hidden_size': 128, 'num_layers': 2}
"lstm":         {'hidden_size': 128, 'num_layers': 4, 'bidirectional': True, 'monolithic': True, 'dropout': 0.5, 'batch_first': True}
"num_channels": 1
"sample_rate":  16000
"sincnet":      {'stride': 10, 'sample_rate': 16000}
```

## Papers

- [pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe](https://hal.science/hal-04247212/document)
- [pyannote.audio speaker diarization pipeline at VoxSRC 2023](https://mmai.io/datasets/voxceleb/voxsrc/data_workshop_2023/reports/pyannote_report.pdf)

