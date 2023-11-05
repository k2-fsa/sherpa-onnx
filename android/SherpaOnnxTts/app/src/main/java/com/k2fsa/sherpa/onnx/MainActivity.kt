package com.k2fsa.sherpa.onnx

import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.io.File

const val TAG = "sherpa-onnx"

class MainActivity : AppCompatActivity() {
    private lateinit var tts: OfflineTts
    private lateinit var text: EditText
    private lateinit var sid: EditText
    private lateinit var speed: EditText
    private lateinit var generate: Button
    private lateinit var play: Button
    private var hasFile: Boolean = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.i(TAG, "Start to initialize TTS")
        initTts()
        Log.i(TAG, "Finish initializing TTS")

        text = findViewById(R.id.text)
        sid = findViewById(R.id.sid)
        speed = findViewById(R.id.speed)

        generate = findViewById(R.id.generate)
        play = findViewById(R.id.play)

        generate.setOnClickListener { onClickGenerate() }
        play.setOnClickListener { onClickPlay() }

        sid.setText("0")
        speed.setText("1.0")

        // we will change sampleText here in the CI
        val sampleText = ""
        text.setText(sampleText)

        play.isEnabled = false;
    }

    fun onClickGenerate() {
        val sidInt = sid.text.toString().toIntOrNull()
        if (sidInt == null || sidInt < 0) {
            Toast.makeText(
                applicationContext,
                "Please input a non-negative integer for speaker ID!",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val speedFloat = speed.text.toString().toFloatOrNull()
        if (speedFloat == null || speedFloat <= 0) {
            Toast.makeText(
                applicationContext,
                "Please input a positive number for speech speed!",
                Toast.LENGTH_SHORT
            ).show()
            return
        }

        val textStr = text.text.toString().trim()
        if (textStr.isBlank() || textStr.isEmpty()) {
            Toast.makeText(applicationContext, "Please input a non-empty text!", Toast.LENGTH_SHORT)
                .show()
            return
        }

        play.isEnabled = false;
        val audio = tts.generate(text = textStr, sid = sidInt, speed = speedFloat)

        val filename = application.filesDir.absolutePath + "/generated.wav"
        val ok = audio.samples.size > 0 && audio.save(filename)
        if (ok) {
            play.isEnabled = true
            // Play automatically after generation
            onClickPlay()
        }
    }

    fun onClickPlay() {
        val filename = application.filesDir.absolutePath + "/generated.wav"
        val mediaPlayer = MediaPlayer.create(
            applicationContext,
            Uri.fromFile(File(filename))
        )
        mediaPlayer.start()
    }

    fun initTts() {
        var modelDir :String?
        var modelName :String?
        var ruleFsts: String?

        // The purpose of such a design is to make the CI test easier
        // Please see
        // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/apk/generate-tts-apk-script.py
        modelDir = null
        modelName = null
        ruleFsts = null

        // Example 1:
        // modelDir = "vits-vctk"
        // modelName = "vits-vctk.onnx"

        // Example 2:
        // modelDir = "vits-piper-en_US-lessac-medium"
        // modelName = "en_US-lessac-medium.onnx"

        // Example 3:
        // modelDir = "vits-zh-aishell3"
        // modelName = "vits-aishell3.onnx"
        // ruleFsts = "vits-zh-aishell3/rule.fst"

        val config = getOfflineTtsConfig(modelDir = modelDir!!, modelName = modelName!!, ruleFsts = ruleFsts ?: "")!!
        tts = OfflineTts(assetManager = application.assets, config = config)
    }
}
