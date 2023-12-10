package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager
import android.media.*
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

const val TAG = "sherpa-onnx"

class MainActivity : AppCompatActivity() {
    private lateinit var tts: OfflineTts
    private lateinit var text: EditText
    private lateinit var sid: EditText
    private lateinit var speed: EditText
    private lateinit var generate: Button
    private lateinit var play: Button

    // see
    // https://developer.android.com/reference/kotlin/android/media/AudioTrack
    private lateinit var track: AudioTrack

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.i(TAG, "Start to initialize TTS")
        initTts()
        Log.i(TAG, "Finish initializing TTS")

        Log.i(TAG, "Start to initialize AudioTrack")
        initAudioTrack()
        Log.i(TAG, "Finish initializing AudioTrack")

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

        play.isEnabled = false
    }

    private fun initAudioTrack() {
        val sampleRate = tts.sampleRate()
        val bufLength = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT
        )
        Log.i(TAG, "sampleRate: ${sampleRate}, buffLength: ${bufLength}")

        val attr = AudioAttributes.Builder().setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .build()

        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
            .setSampleRate(sampleRate)
            .build()

        track = AudioTrack(
            attr, format, bufLength, AudioTrack.MODE_STREAM,
            AudioManager.AUDIO_SESSION_ID_GENERATE
        )
        track.play()
    }

    // this function is called from C++
    private fun callback(samples: FloatArray) {
        track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
    }

    private fun onClickGenerate() {
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

        track.pause()
        track.flush()
        track.play()

        play.isEnabled = false
        Thread {
            val audio = tts.generateWithCallback(
                text = textStr,
                sid = sidInt,
                speed = speedFloat,
                callback = this::callback
            )

            val filename = application.filesDir.absolutePath + "/generated.wav"
            val ok = audio.samples.size > 0 && audio.save(filename)
            if (ok) {
                runOnUiThread {
                    play.isEnabled = true
                    track.stop()
                }
            }
        }.start()
    }

    private fun onClickPlay() {
        val filename = application.filesDir.absolutePath + "/generated.wav"
        val mediaPlayer = MediaPlayer.create(
            applicationContext,
            Uri.fromFile(File(filename))
        )
        mediaPlayer.start()
    }

    private fun initTts() {
        var modelDir: String?
        var modelName: String?
        var ruleFsts: String?
        var lexicon: String?
        var dataDir: String?
        var assets: AssetManager? = application.assets

        // The purpose of such a design is to make the CI test easier
        // Please see
        // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/apk/generate-tts-apk-script.py
        modelDir = null
        modelName = null
        ruleFsts = null
        lexicon = null
        dataDir = null

        // Example 1:
        // modelDir = "vits-vctk"
        // modelName = "vits-vctk.onnx"
        // lexicon = "lexicon.txt"

        // Example 2:
        // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
        // modelDir = "vits-piper-en_US-amy-low"
        // modelName = "en_US-amy-low.onnx"
        // dataDir = "vits-piper-en_US-amy-low/espeak-ng-data"

        // Example 3:
        // modelDir = "vits-zh-aishell3"
        // modelName = "vits-aishell3.onnx"
        // ruleFsts = "vits-zh-aishell3/rule.fst"
        // lexcion = "lexicon.txt"

        if (dataDir != null) {
            val newDir = copyDataDir(modelDir)
            modelDir = newDir + "/" + modelDir
            dataDir = newDir + "/" + dataDir
            assets = null
        }

        val config = getOfflineTtsConfig(
            modelDir = modelDir!!, modelName = modelName!!, lexicon = lexicon ?: "",
            dataDir = dataDir ?: "",
            ruleFsts = ruleFsts ?: ""
        )!!

        tts = OfflineTts(assetManager = assets, config = config)
    }


    private fun copyDataDir(dataDir: String): String {
        println("data dir is $dataDir")
        copyAssets(dataDir)

        val newDataDir = application.getExternalFilesDir(null)!!.absolutePath
        println("newDataDir: $newDataDir")
        return newDataDir
    }

    private fun copyAssets(path: String) {
        val assets: Array<String>?
        try {
            assets = application.assets.list(path)
            if (assets!!.isEmpty()) {
                copyFile(path)
            } else {
                val fullPath = "${application.getExternalFilesDir(null)}/$path"
                val dir = File(fullPath)
                dir.mkdirs()
                for (asset in assets.iterator()) {
                    val p: String = if (path == "") "" else path + "/"
                    copyAssets(p + asset)
                }
            }
        } catch (ex: IOException) {
            Log.e(TAG, "Failed to copy $path. ${ex.toString()}")
        }
    }

    private fun copyFile(filename: String) {
        try {
            val istream = application.assets.open(filename)
            val newFilename = application.getExternalFilesDir(null).toString() + "/" + filename
            val ostream = FileOutputStream(newFilename)
            // Log.i(TAG, "Copying $filename to $newFilename")
            val buffer = ByteArray(1024)
            var read = 0
            while (read != -1) {
                ostream.write(buffer, 0, read)
                read = istream.read(buffer)
            }
            istream.close()
            ostream.flush()
            ostream.close()
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to copy $filename, ${ex.toString()}")
        }
    }
}
