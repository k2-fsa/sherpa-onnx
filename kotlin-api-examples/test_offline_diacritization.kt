package com.k2fsa.sherpa.onnx

fun main() {
  testDiacritization()
}

fun testDiacritization() {
  // please download the model from
  // https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip
  val config = OfflineDiacritizationConfig(
      model = OfflineDiacritizationModelConfig(
          cattEncoder = "./catt_eo_model_onnx/encoder.onnx",
          cattDecoder = "./catt_eo_model_onnx/decoder.onnx",
          numThreads = 1,
          debug = true,
          provider = "cpu",
      )
  )
  val diacrt = OfflineDiacritization(config = config)
  val sentences = arrayOf(
      "وقالت مجلة نيوزويك الأمريكية التحديث الجديد ل إنستجرام يمكن أن يساهم في إيقاف وكشف الحسابات المزورة بسهولة شديدة",
      "اللغة العربية من أقدم اللغات السامية",
  )
  println("---")
  for (text in sentences) {
    val out = diacrt.addDiacritics(text)
    println("Input: $text")
    println("Output: $out")
    println("---")
  }

  diacrt.release()
}
