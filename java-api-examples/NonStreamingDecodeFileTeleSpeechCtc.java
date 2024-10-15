// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline TeleSpeech CTC model
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileTeleSpeechCtc {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/telespeech/models.html#sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04
    // to download model files
    String model = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/model.int8.onnx";
    String tokens = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/tokens.txt";

    String waveFilename = "./sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04/test_wavs/3-sichuan.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setTeleSpeech(model)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .setModelType("telespeech_ctc")
            .build();

    OfflineRecognizerConfig config =
        OfflineRecognizerConfig.builder()
            .setOfflineModelConfig(modelConfig)
            .setDecodingMethod("greedy_search")
            .build();

    OfflineRecognizer recognizer = new OfflineRecognizer(config);
    OfflineStream stream = recognizer.createStream();
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    recognizer.decode(stream);

    String text = recognizer.getResult(stream).getText();

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

    stream.release();
    recognizer.release();
  }
}
