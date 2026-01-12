// Copyright 2026 Xiaomi Corporation

// This file shows how to use an offline FunASR Nano model,
// i.e., non-streaming FunASR Nano model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileFunAsrNano {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/funasr-nano/index.html
    // to download model files
    String encoderAdaptor = "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx";
    String llm = "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx";
    String embedding = "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx";
    String tokenizer = "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B";

    String tokens = "";

    String waveFilename = "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineFunAsrNanoModelConfig funasrNano =
        OfflineFunAsrNanoModelConfig.builder()
            .setEncoderAdaptor(encoderAdaptor)
            .setLLM(llm)
            .setEmbedding(embedding)
            .setTokenizer(tokenizer)
            .build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setFunAsrNano(funasrNano)
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
