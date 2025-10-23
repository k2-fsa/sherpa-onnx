# Real-time speech recognition by non streaming and VAD

This APP supports the following platforms:

- macOS (tested)

## Getting Started

Follow these steps to download and set up the required models to run the demo successfully.

### 1. Select a non-streaming model

Choose one of the following non-streaming ASR models:

#### Code Available Models:
- **whisper**: Whisper base model
- **senseVoice**: SenseVoice multilingual model (supports Chinese, English, Japanese, Korean, Cantonese)
- **parakeet-tdt**: NeMo transducer-based parakeet-tdt model

#### Model Download Links:
- **whisper**: https://huggingface.co/csukuangfj/sherpa-onnx-whisper-base
- **senseVoice**: https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09  
- **parakeet-tdt**: https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8

### 2. Download VAD Model

Download the VAD (Voice Activity Detection) model from:
https://huggingface.co/csukuangfj/vad

Place the VAD model file (e.g., `silero_vad.onnx`) in the `assets` directory.

### 3. Configure the Model in Code

#### Step 3.1: Update Model Selection
Edit `lib/non_streaming_vad_asr.dart` and set the model type:

```dart
Future<sherpa_onnx.OfflineRecognizer> createOfflineRecognizer() async {
  final type = 2; // 0: whisper, 1: senseVoice, 2: parakeet-tdt
  final modelConfig = await getOfflineModelConfig(type: type);
  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  return sherpa_onnx.OfflineRecognizer(config);
}
```

#### Step 3.2: Update Asset Configuration
Edit `pubspec.yaml` and add the appropriate asset directory for your chosen model:

```yaml
flutter:
  assets:
    - assets/
    - assets/whisper/        # For whisper model
    # - assets/senseVoice/    # For senseVoice model (uncomment when using)
    # - assets/nemo_transducer/ # For parakeet-tdt model (uncomment when using)
```

### 4. Directory Structure Setup

#### For whisper model:
```
./assets/
├── whisper/
│   ├── base-decoder.onnx
│   ├── base-encoder.onnx
│   └── base-tokens.txt
└── silero_vad.onnx
```

#### For senseVoice model:
```
./assets/
├── senseVoice/
│   ├── model.int8.onnx
│   └── tokens.txt
└── silero_vad.onnx
```

#### For parakeet-tdt model:
```
./assets/
├── nemo_transducer/
│   ├── encoder.int8.onnx
│   ├── decoder.int8.onnx
│   ├── joiner.int8.onnx
│   └── tokens.txt
└── silero_vad.onnx
```

### 5. Advanced Configuration (Optional)

#### Modify Model Configuration:
You can edit `lib/offline_model.dart` to customize the model configuration, such as model size and quantization settings.

#### Adjust Audio Recording Settings:
In `lib/non_streaming_vad_asr.dart`, you can modify the VAD configuration:

```dart
_vad = sherpa_onnx.VoiceActivityDetector(
  config: _vadConfig, 
  bufferSizeInSeconds: 30  // Adjust based on your needs
);
_buffer = sherpa_onnx.CircularBuffer(capacity: 30 * 16000);
```

### 6. Run the Application

Use the following command to run the app:

```bash
flutter run -d macos
```

## Troubleshooting

- Ensure all model files are placed in the correct directories
- Check that `pubspec.yaml` includes the correct asset paths
- Verify the model type in `non_streaming_vad_asr.dart` matches your chosen model
- Make sure to delete unnecessary files to reduce app size

## Notes

- The VAD model is required for all non-streaming ASR models
- Model performance may vary depending on hardware capabilities
- Adjust buffer sizes and VAD parameters based on your specific use case
