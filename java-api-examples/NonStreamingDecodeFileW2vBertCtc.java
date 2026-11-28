// Copyright 2026 graehl

// This file shows how to use an offline w2v-BERT 2.0 CTC model,
// i.e., non-streaming w2v-BERT 2.0 CTC model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileW2vBertCtc {
  public static void main(String[] args) {
    // please download model files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20.tar.bz2
    String model = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/model.int8.onnx";

    String tokens = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/tokens.txt";

    String waveFilename = "./sherpa-onnx-w2v-bert-ctc-gl-int8-2026-11-20/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineW2vBertCtcModelConfig w2vBertCtc =
        OfflineW2vBertCtcModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setW2vBertCtc(w2vBertCtc)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
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
