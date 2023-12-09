package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

fun callback(samples: FloatArray): Unit {
  println("callback got called with ${samples.size} samples");
}

fun main() {
  testTts()
  testAsr()
}

fun testTts() {
  // see https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  var config = OfflineTtsConfig(
    model=OfflineTtsModelConfig(
      vits=OfflineTtsVitsModelConfig(
        model="./vits-piper-en_US-amy-low/en_US-amy-low.onnx",
        tokens="./vits-piper-en_US-amy-low/tokens.txt",
        dataDir="./vits-piper-en_US-amy-low/espeak-ng-data",
      ),
      numThreads=1,
      debug=true,
    )
  )
  val tts = OfflineTts(config=config)
  val audio = tts.generateWithCallback(text="“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.”", callback=::callback)
  audio.save(filename="test-en.wav")
}

fun testAsr() {
    var featConfig = FeatureConfig(
        sampleRate = 16000,
        featureDim = 80,
    )

    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
    // to dowload pre-trained models
    var modelConfig = OnlineModelConfig(
        transducer = OnlineTransducerModelConfig(
            encoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx",
            decoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx",
            joiner = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx",
        ),
        tokens = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt",
        numThreads = 1,
        debug = false,
    )

    var endpointConfig = EndpointConfig()

    var lmConfig = OnlineLMConfig()

    var config = OnlineRecognizerConfig(
        modelConfig = modelConfig,
        lmConfig = lmConfig,
        featConfig = featConfig,
        endpointConfig = endpointConfig,
        enableEndpoint = true,
        decodingMethod = "greedy_search",
        maxActivePaths = 4,
    )

    var model = SherpaOnnx(
        config = config,
    )

    var objArray = WaveReader.readWaveFromFile(
        filename = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/0.wav",
    )
    var samples: FloatArray = objArray[0] as FloatArray
    var sampleRate: Int = objArray[1] as Int

    model.acceptWaveform(samples, sampleRate = sampleRate)
    while (model.isReady()) {
        model.decode()
    }

    var tailPaddings = FloatArray((sampleRate * 0.5).toInt()) // 0.5 seconds
    model.acceptWaveform(tailPaddings, sampleRate = sampleRate)
    model.inputFinished()
    while (model.isReady()) {
        model.decode()
    }

    println("results: ${model.text}")
}
