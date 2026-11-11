// Copyright 2026 Xiaomi Corporation

// This file shows how to use a non-streaming Zipformer CTC model with an HLG
// decoding graph via sherpa-onnx's Java API.
//
// The HLG graph is passed to the C API through CtcFstDecoderConfig.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingZipformerCtcHlg {
  public static void main(String[] args) {
    // please refer to
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
    // to download model files
    String dir = "./sherpa-onnx-zipformer-ctc-en-2023-10-02";
    String model = dir + "/model.int8.onnx";
    String tokens = dir + "/tokens.txt";
    String graph = dir + "/HLG.fst";
    String waveFilename = dir + "/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineZipformerCtcModelConfig zipformerCtc =
        OfflineZipformerCtcModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setZipformerCtc(zipformerCtc)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    // This is what the example exercises: the graph and max_active reach the C
    // API through OfflineCtcFstDecoderConfig.
    OfflineCtcFstDecoderConfig ctcFstDecoderConfig =
        OfflineCtcFstDecoderConfig.builder()
            .setGraph(graph)
            .setMaxActive(3000)
            .build();

    OfflineRecognizerConfig config =
        OfflineRecognizerConfig.builder()
            .setOfflineModelConfig(modelConfig)
            .setDecodingMethod("greedy_search")
            .setCtcFstDecoderConfig(ctcFstDecoderConfig)
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
