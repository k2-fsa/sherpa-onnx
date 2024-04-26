// Copyright 2024 Xiaomi Corporation

// This file shows how to use a punctuation model to add punctuations to text.
//
// The model supports both English and Chinese.
import com.k2fsa.sherpa.onnx.*;

public class AddPunctuation {
  public static void main(String[] args) {
    // please download the model from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
    String model = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx";
    OfflinePunctuationModelConfig modelConfig =
        OfflinePunctuationModelConfig.builder()
            .setCtTransformer(model)
            .setNumThreads(1)
            .setDebug(true)
            .build();
    OfflinePunctuationConfig config =
        OfflinePunctuationConfig.builder().setModel(modelConfig).build();

    OfflinePunctuation punct = new OfflinePunctuation(config);

    String[] sentences =
        new String[] {
          "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
          "我们都是木头人不会说话不会动",
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
