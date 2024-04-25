// Copyright 2024 Xiaomi Corporation

// This file shows how to use an online CTC model, i.e., streaming CTC model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class StreamingDecodeFileCtcHLG {
  public static void main(String[] args) {
    // please refer to
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
    // to download model files
    String model =
        "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx";
    String tokens = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt";
    String hlg = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst";
    String waveFilename = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OnlineZipformer2CtcModelConfig ctc =
        OnlineZipformer2CtcModelConfig.builder().setModel(model).build();

    OnlineModelConfig modelConfig =
        OnlineModelConfig.builder()
            .setZipformer2Ctc(ctc)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OnlineCtcFstDecoderConfig ctcFstDecoderConfig =
        OnlineCtcFstDecoderConfig.builder().setGraph("hlg").build();

    OnlineRecognizerConfig config =
        OnlineRecognizerConfig.builder()
            .setOnlineModelConfig(modelConfig)
            .setCtcFstDecoderConfig(ctcFstDecoderConfig)
            .build();

    OnlineRecognizer recognizer = new OnlineRecognizer(config);
    OnlineStream stream = recognizer.createStream();
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    float[] tailPaddings = new float[(int) (0.3 * reader.getSampleRate())];
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
