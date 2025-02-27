// Copyright 2025 Xiaomi Corporation

// This file shows how to use a punctuation model to add punctuations to text.
//
// The model supports both English and Chinese.
import com.k2fsa.sherpa.onnx.*;

public class OnlineAddPunctuation {
  public static void main(String[] args) {
    // please download the model from
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
    String model = "./sherpa-onnx-online-punct-en-2024-08-06/model.int8.onnx";
    String bpeVocab = "./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab";
    OnlinePunctuationModelConfig modelConfig =
        OnlinePunctuationModelConfig.builder()
            .setCnnBilstm(model)
            .setBpeVocab(bpeVocab)
            .setNumThreads(1)
            .setDebug(true)
            .build();
    OnlinePunctuationConfig config =
        OnlinePunctuationConfig.builder().setModel(modelConfig).build();

    OnlinePunctuation punct = new OnlinePunctuation(config);

    String[] sentences =
        new String[] {
          "how are you doing fantastic thank you how about you",
          "The African blogosphere is rapidly expanding bringing more voices online in the form of"
              + " commentaries opinions analyses rants and poetry",
        };

    System.out.println("---");
    for (String text : sentences) {
      String out = punct.addPunctuation(text);
      System.out.printf("Input: %s\n", text);
      System.out.printf("Output: %s\n", out);
      System.out.println("---");
    }
  }
}
