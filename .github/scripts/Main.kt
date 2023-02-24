package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

fun main() {
    var featConfig = FeatureConfig(
        sampleRate = 16000.0f,
        featureDim = 80,
    )

    var modelConfig = OnlineTransducerModelConfig(
        encoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/encoder-epoch-99-avg-1.onnx",
        decoder = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/decoder-epoch-99-avg-1.onnx",
        joiner = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/joiner-epoch-99-avg-1.onnx",
        tokens = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt",
        numThreads = 4,
        debug = false,
    )

    var endpointConfig = EndpointConfig()

    var config = OnlineRecognizerConfig(
        modelConfig = modelConfig,
        featConfig = featConfig,
        endpointConfig = endpointConfig,
        enableEndpoint = true,
    )

    var model = SherpaOnnx(
        assetManager = AssetManager(),
        config = config,
    )
    var samples = WaveReader.readWave(
        assetManager = AssetManager(),
        filename = "./sherpa-onnx-streaming-zipformer-en-2023-02-21/test_wavs/1089-134686-0001.wav",
    )

    model.decodeSamples(samples!!)

    var tail_paddings = FloatArray(8000) // 0.5 seconds
    model.decodeSamples(tail_paddings)

    model.inputFinished()
    println("results: ${model.text}")
}
