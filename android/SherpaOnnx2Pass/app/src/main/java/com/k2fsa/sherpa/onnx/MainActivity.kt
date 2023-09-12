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
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import kotlin.concurrent.thread

private const val TAG = "sherpa-onnx"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

class MainActivity : AppCompatActivity() {
    private val permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    private lateinit var onlineRecognizer: SherpaOnnx
    private lateinit var offlineRecognizer: SherpaOnnxOffline
    private var audioRecord: AudioRecord? = null
    private lateinit var recordButton: Button
    private lateinit var textView: TextView
    private var recordingThread: Thread? = null

    private val audioSource = MediaRecorder.AudioSource.MIC
    private val sampleRateInHz = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO

    private var samplesBuffer = arrayListOf<FloatArray>()

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

        Log.i(TAG, "Start to initialize first-pass recognizer")
        initOnlineRecognizer()
        Log.i(TAG, "Finished initializing first-pass recognizer")

        Log.i(TAG, "Start to initialize second-pass recognizer")
        initOfflineRecognizer()
        Log.i(TAG, "Finished initializing second-pass recognizer")

        recordButton = findViewById(R.id.record_button)
        recordButton.setOnClickListener { onclick() }

        textView = findViewById(R.id.my_text)
        textView.movementMethod = ScrollingMovementMethod()
    }

    private fun onclick() {
        if (!isRecording) {
            val ret = initMicrophone()
            if (!ret) {
                Log.e(TAG, "Failed to initialize microphone")
                return
            }
            Log.i(TAG, "state: ${audioRecord?.state}")
            audioRecord!!.startRecording()
            recordButton.setText(R.string.stop)
            isRecording = true
            onlineRecognizer.reset(true)
            samplesBuffer.clear()
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
                samplesBuffer.add(samples)

                onlineRecognizer.acceptWaveform(samples, sampleRate = sampleRateInHz)
                while (onlineRecognizer.isReady()) {
                    onlineRecognizer.decode()
                }
                val isEndpoint = onlineRecognizer.isEndpoint()
                var textToDisplay = lastText

                var text = onlineRecognizer.text
                if (text.isNotBlank()) {
                    if (lastText.isBlank()) {
                        // textView.text = "${idx}: ${text}"
                        textToDisplay = "${idx}: ${text}"
                    } else {
                        textToDisplay = "${lastText}\n${idx}: ${text}"
                    }
                }

                if (isEndpoint) {
                    onlineRecognizer.reset()

                    if (text.isNotBlank()) {
                        text = runSecondPass()
                        lastText = "${lastText}\n${idx}: ${text}"
                        idx += 1
                    } else {
                        samplesBuffer.clear()
                    }
                }

                runOnUiThread {
                    textView.text = textToDisplay.lowercase()
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

    private fun initOnlineRecognizer() {
        // Please change getModelConfig() to add new models
        // See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
        // for a list of available models
        val firstType = 1
        println("Select model type ${firstType} for the first pass")
        val config = OnlineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRate = sampleRateInHz, featureDim = 80),
            modelConfig = getModelConfig(type = firstType)!!,
            endpointConfig = getEndpointConfig(),
            enableEndpoint = true,
        )

        onlineRecognizer = SherpaOnnx(
            assetManager = application.assets,
            config = config,
        )
    }

    private fun initOfflineRecognizer() {
        // Please change getOfflineModelConfig() to add new models
        // See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
        // for a list of available models
        val secondType = 1
        println("Select model type ${secondType} for the second pass")

        val config = OfflineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRate = sampleRateInHz, featureDim = 80),
            modelConfig = getOfflineModelConfig(type = secondType)!!,
        )

        offlineRecognizer = SherpaOnnxOffline(
            assetManager = application.assets,
            config = config,
        )
    }

    private fun runSecondPass(): String {
        var totalSamples = 0
        for (a in samplesBuffer) {
            totalSamples += a.size
        }
        var i = 0

        val samples = FloatArray(totalSamples)

        // todo(fangjun): Make it more efficient
        for (a in samplesBuffer) {
            for (s in a) {
                samples[i] = s
                i += 1
            }
        }


        val n = maxOf(0, samples.size - 8000)

        samplesBuffer.clear()
        samplesBuffer.add(samples.sliceArray(n..samples.size-1))

        return offlineRecognizer.decode(samples.sliceArray(0..n), sampleRateInHz)
    }
}
