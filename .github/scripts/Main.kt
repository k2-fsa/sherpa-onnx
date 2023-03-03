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

    var samples = WaveReader.readWave(
        assetManager = AssetManager(),
        filename = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/1089-134686-0001.wav",
    )

    model.acceptWaveform(samples!!, sampleRate=16000)
    while (model.isReady()) {
      model.decode()
    }

    var tail_paddings = FloatArray(8000) // 0.5 seconds
    model.acceptWaveform(tail_paddings, sampleRate=16000)
    model.inputFinished()
    while (model.isReady()) {
      model.decode()
    }

    println("results: ${model.text}")
}
