package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        println("Started!")
        testDecodeWave()
    }

    fun testDecodeWave() {
        val config = OnlineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRate = 16000.0f, featureDim = 80),
            modelConfig = getModelConfig(type=0)!!,
            endpointConfig = getEndpointConfig(),
            enableEndpoint = true
        )

        val model = SherpaOnnx(
            assetManager = application.assets,
            config = config,
        )
        println("Read model done!")

        val samples = WaveReader.readWave(
            assetManager = application.assets,
            filename = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav",
        )

        model.decodeSamples(samples!!)

        val tailPaddings = FloatArray(8000) // 0.5 seconds
        model.decodeSamples(tailPaddings)

        model.inputFinished()
        println(model.text)
    }
}