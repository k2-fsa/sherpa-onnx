package com.k2fsa.sherpa.onnx

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.k2fsa.sherpa.onnx.*
import kotlin.concurrent.thread

private const val TAG = "sherpa-onnx"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

class MainActivity : AppCompatActivity() {
    private val permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    private lateinit var model: SherpaOnnxKws
    private var audioRecord: AudioRecord? = null
    private lateinit var recordButton: Button
    private lateinit var textView: TextView
    private lateinit var inputText: EditText
    private var recordingThread: Thread? = null

    private val audioSource = MediaRecorder.AudioSource.MIC
    private val sampleRateInHz = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO

    // Note: We don't use AudioFormat.ENCODING_PCM_FLOAT
    // since the AudioRecord.read(float[]) needs API level >= 23
    // but we are targeting API level >= 21
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private var idx: Int = 0
    private var lastText: String = ""

    @Volatile
    private var isRecording: Boolean = false

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val permissionToRecordAccepted = if (requestCode == REQUEST_RECORD_AUDIO_PERMISSION) {
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        } else {
            false
        }

        if (!permissionToRecordAccepted) {
            Log.e(TAG, "Audio record is disallowed")
            finish()
        }

        Log.i(TAG, "Audio record is permitted")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)

        Log.i(TAG, "Start to initialize model")
        initModel()
        Log.i(TAG, "Finished initializing model")

        recordButton = findViewById(R.id.record_button)
        recordButton.setOnClickListener { onclick() }

        textView = findViewById(R.id.my_text)
        textView.movementMethod = ScrollingMovementMethod()

        inputText = findViewById(R.id.input_text)
    }

    private fun onclick() {
        if (!isRecording) {
            var keywords = inputText.text.toString()

            Log.i(TAG, keywords)
            keywords = keywords.replace("\n", "/")
            // If keywords is an empty string, it just resets the decoding stream
            // always returns true in this case.
            // If keywords is not empty, it will create a new decoding stream with
            // the given keywords appended to the default keywords.
            // Return false if errors occured when adding keywords, true otherwise.
            val status = model.reset(keywords)
            if (!status) {
                Log.i(TAG, "Failed to reset with keywords.")
                Toast.makeText(this, "Failed to set keywords.", Toast.LENGTH_LONG).show();
                return
            }

            val ret = initMicrophone()
            if (!ret) {
                Log.e(TAG, "Failed to initialize microphone")
                return
            }
            Log.i(TAG, "state: ${audioRecord?.state}")
            audioRecord!!.startRecording()
            recordButton.setText(R.string.stop)
            isRecording = true
            textView.text = ""
            lastText = ""
            idx = 0

            recordingThread = thread(true) {
                processSamples()
            }
            Log.i(TAG, "Started recording")
        } else {
            isRecording = false
            audioRecord!!.stop()
            audioRecord!!.release()
            audioRecord = null
            recordButton.setText(R.string.start)
            Log.i(TAG, "Stopped recording")
        }
    }

    private fun processSamples() {
        Log.i(TAG, "processing samples")

        val interval = 0.1 // i.e., 100 ms
        val bufferSize = (interval * sampleRateInHz).toInt() // in samples
        val buffer = ShortArray(bufferSize)

        while (isRecording) {
            val ret = audioRecord?.read(buffer, 0, buffer.size)
            if (ret != null && ret > 0) {
                val samples = FloatArray(ret) { buffer[it] / 32768.0f }
                model.acceptWaveform(samples, sampleRate=sampleRateInHz)
                while (model.isReady()) {
                    model.decode()
                }

                val text = model.keyword

                var textToDisplay = lastText;

                if(text.isNotBlank()) {
                    if (lastText.isBlank()) {
                        textToDisplay = "${idx}: ${text}"
                    } else {
                        textToDisplay = "${idx}: ${text}\n${lastText}"
                    }
                    lastText = "${idx}: ${text}\n${lastText}"
                    idx += 1
                }

                runOnUiThread {
                    textView.text = textToDisplay
                }
            }
        }
    }

    private fun initMicrophone(): Boolean {
        if (ActivityCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION)
            return false
        }

        val numBytes = AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat)
        Log.i(
            TAG, "buffer size in milliseconds: ${numBytes * 1000.0f / sampleRateInHz}"
        )

        audioRecord = AudioRecord(
            audioSource,
            sampleRateInHz,
            channelConfig,
            audioFormat,
            numBytes * 2 // a sample has two bytes as we are using 16-bit PCM
        )
        return true
    }

    private fun initModel() {
        // Please change getModelConfig() to add new models
        // See https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
        // for a list of available models
        val type = 0
        Log.i(TAG, "Select model type ${type}")
        val config = KeywordSpotterConfig(
            featConfig = getFeatureConfig(sampleRate = sampleRateInHz, featureDim = 80),
            modelConfig = getModelConfig(type = type)!!,
            keywordsFile = getKeywordsFile(type = type)!!,
        )

        model = SherpaOnnxKws(
            assetManager = application.assets,
            config = config,
        )
    }
}
