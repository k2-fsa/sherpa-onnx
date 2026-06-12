// Copyright 2026 Matias Lin

// This file shows how to use a CATT diacritization model to add diacritics to Arabic text.
import com.k2fsa.sherpa.onnx.*;

public class OfflineAddDiacritics {
  public static void main(String[] args) {
    // please download the model from
    // https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
    String cattEncoder = "./catt_eo_model_onnx/encoder.onnx";
    String cattDecoder = "./catt_eo_model_onnx/decoder.onnx";
    OfflineDiacritizationModelConfig modelConfig =
        OfflineDiacritizationModelConfig.builder()
            .setCattEncoder(cattEncoder)
            .setCattDecoder(cattDecoder)
            .setNumThreads(1)
            .setDebug(true)
            .build();
    OfflineDiacritizationConfig config =
        OfflineDiacritizationConfig.builder().setModel(modelConfig).build();

    OfflineDiacritization diacrt = new OfflineDiacritization(config);

    String[] sentences =
        new String[] {
          "وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة",
          "اللغة العربية من أقدم اللغات السامية",
        };

    System.out.println("---");
    for (String text : sentences) {
      String out = diacrt.addDiacritics(text);
      System.out.printf("Input: %s\n", text);
      System.out.printf("Output: %s\n", out);
      System.out.println("---");
    }
  }
}
