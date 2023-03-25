package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

fun main() {
    var featConfig = FeatureConfig(
        sampleRate = 16000,
        featureDim = 80,
    )

    var modelConfig = OnlineTransducerModelConfig(
        encoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx",
        decoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx",
        joiner = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx",
        tokens = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt",
        numThreads = 1,
        debug = false,
    )

    var endpointConfig = EndpointConfig()

    var config = OnlineRecognizerConfig(
        modelConfig = modelConfig,
        featConfig = featConfig,
        endpointConfig = endpointConfig,
        enableEndpoint = true,
        decodingMethod = "greedy_search",
        maxActivePaths = 4,
    )

    var model = SherpaOnnx(
        assetManager = AssetManager(),
        config = config,
    )

    var objArray = WaveReader.readWave(
        assetManager = AssetManager(),
        filename = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/1089-134686-0001.wav",
    )
    var samples : FloatArray = objArray[0] as FloatArray
    var sampleRate : Int = objArray[1] as Int

    model.acceptWaveform(samples, sampleRate=sampleRate)
    while (model.isReady()) {
      model.decode()
    }

    var tail_paddings = FloatArray((sampleRate * 0.5).toInt()) // 0.5 seconds
    model.acceptWaveform(tail_paddings, sampleRate=sampleRate)
    model.inputFinished()
    while (model.isReady()) {
      model.decode()
    }

    println("results: ${model.text}")
}
