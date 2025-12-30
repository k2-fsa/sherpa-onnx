// Copyright 2024 Xiaomi Corporation

// This file shows how to use an online T-one CTC model, i.e.,
// streaming T-one CTC model, to decode files.
import com.k2fsa.sherpa.onnx.*;

public class StreamingDecodeFileToneCtc {
  public static void main(String[] args) {
    String model = "./sherpa-onnx-streaming-t-one-russian-2025-09-08/model.onnx";
    String tokens = "./sherpa-onnx-streaming-t-one-russian-2025-09-08/tokens.txt";
    String waveFilename = "./sherpa-onnx-streaming-t-one-russian-2025-09-08/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OnlineToneCtcModelConfig ctc = OnlineToneCtcModelConfig.builder().setModel(model).build();

    OnlineModelConfig modelConfig =
        OnlineModelConfig.builder()
            .setToneCtc(ctc)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OnlineRecognizerConfig config =
        OnlineRecognizerConfig.builder()
            .setOnlineModelConfig(modelConfig)
            .setDecodingMethod("greedy_search")
            .build();

    OnlineRecognizer recognizer = new OnlineRecognizer(config);
    OnlineStream stream = recognizer.createStream();

    float[] leftPaddings = new float[(int) (0.3 * reader.getSampleRate())];
    stream.acceptWaveform(leftPaddings, reader.getSampleRate());

    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    float[] tailPaddings = new float[(int) (0.6 * reader.getSampleRate())];
    stream.acceptWaveform(tailPaddings, reader.getSampleRate());

    while (recognizer.isReady(stream)) {
      recognizer.decode(stream);
    }

    String text = recognizer.getResult(stream).getText();

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

    stream.release();
    recognizer.release();
  }
}
