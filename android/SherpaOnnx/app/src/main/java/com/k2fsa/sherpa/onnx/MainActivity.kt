package com.k2fsa.sherpa.onnx

import android.Manifest
import android.content.Context
import android.content.res.AssetManager
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
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.concurrent.thread

private const val TAG = "sherpa-onnx"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

// To enable microphone in android emulator, use
//
// adb emu avd hostmicon

private fun assetExists(assetManager: AssetManager, path: String): Boolean {
    val dir = path.substringBeforeLast('/', "")
    val fileName = path.substringAfterLast('/')

    val files = assetManager.list(dir) ?: return false
    return files.contains(fileName)
}

private fun copyAssetToInternalStorage(path: String, context: Context): String {
    val targetRoot = context.filesDir
    val outFile = File(targetRoot, path)

    if (!assetExists(context.assets, path = path)) {
        outFile.parentFile?.mkdirs()
        Log.i(TAG, "$path does not exist, return ${outFile.absolutePath}")
        return outFile.absolutePath
    }

    if (outFile.exists()) {
        val assetSize = context.assets.open(path).use { it.available() }
        if (outFile.length() == assetSize.toLong()) {
            Log.i(TAG, "$targetRoot/$path already exists, skip copying, return $targetRoot/$path")
            return outFile.absolutePath
        }
    }

    outFile.parentFile?.mkdirs()

    context.assets.open(path).use { input: InputStream ->
        FileOutputStream(outFile).use { output: OutputStream ->
            input.copyTo(output)
        }
    }
    Log.i(TAG, "Copied $path to $targetRoot/$path")

    return outFile.absolutePath
}

private fun copyAssetListToInternalStorage(paths: String, context: Context): String {
    if (paths.isBlank()) return paths

    return paths.split(",")
        .map { it.trim() }
        .filter { it.isNotEmpty() }
        .map { copyAssetToInternalStorage(it, context) }
        .joinToString(",")
}

class MainActivity : AppCompatActivity() {
    private val permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    private lateinit var recognizer: OnlineRecognizer
    private var audioRecord: AudioRecord? = null
    private lateinit var recordButton: Button
    private lateinit var textView: TextView
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
        val stream = recognizer.createStream()

        val interval = 0.1 // i.e., 100 ms
        val bufferSize = (interval * sampleRateInHz).toInt() // in samples
        val buffer = ShortArray(bufferSize)

        while (isRecording) {
            val ret = audioRecord?.read(buffer, 0, buffer.size)
            if (ret != null && ret > 0) {
                val samples = FloatArray(ret) { buffer[it] / 32768.0f }
                stream.acceptWaveform(samples, sampleRate = sampleRateInHz)
                while (recognizer.isReady(stream)) {
                    recognizer.decode(stream)
                }

                val isEndpoint = recognizer.isEndpoint(stream)
                var text = recognizer.getResult(stream).text

                // For streaming parformer, we need to manually add some
                // paddings so that it has enough right context to
                // recognize the last word of this segment
                if (isEndpoint && recognizer.config.modelConfig.paraformer.encoder.isNotBlank()) {
                    val tailPaddings = FloatArray((0.8 * sampleRateInHz).toInt())
                    stream.acceptWaveform(tailPaddings, sampleRate = sampleRateInHz)
                    while (recognizer.isReady(stream)) {
                        recognizer.decode(stream)
                    }
                    text = recognizer.getResult(stream).text
                }

                var textToDisplay = lastText

                if (text.isNotBlank()) {
                    textToDisplay = if (lastText.isBlank()) {
                        "${idx}: $text"
                    } else {
                        "${lastText}\n${idx}: $text"
                    }
                }

                if (isEndpoint) {
                    recognizer.reset(stream)
                    if (text.isNotBlank()) {
                        lastText = "${lastText}\n${idx}: $text"
                        textToDisplay = lastText
                        idx += 1
                    }
                }

                runOnUiThread {
                    textView.text = textToDisplay
                }
            }
        }
        stream.release()
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
        // See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
        // for a list of available models
        val type = 0
        var ruleFsts : String?
        ruleFsts = null

        val useHr = false
        val hr =  HomophoneReplacerConfig(
            // Used only when useHr is true
            // Please download the following 3 files from
            // https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files
            //
            // dict and lexicon.txt can be shared by different apps
            //
            // replace.fst is specific for an app
            lexicon = "lexicon.txt",
            ruleFsts = "replace.fst",
        )

        Log.i(TAG, "Select model type $type")
        var config = OnlineRecognizerConfig(
            featConfig = getFeatureConfig(sampleRate = sampleRateInHz, featureDim = 80),
            modelConfig = getModelConfig(type = type)!!,
            // lmConfig = getOnlineLMConfig(type = type),
            endpointConfig = getEndpointConfig(),
            enableEndpoint = true,
        )

        if (ruleFsts != null) {
            config.ruleFsts = ruleFsts
        }

        if (useHr) {
            config.hr = hr
        }

        var assetManager: AssetManager? = application.assets
        if (config.modelConfig.provider == "qnn") {
            Log.i(TAG, "nativelibdir: ${applicationInfo.nativeLibraryDir}")
            OnlineRecognizer.prependAdspLibraryPath(applicationInfo.nativeLibraryDir)

            val transducer = config.modelConfig.transducer
            val qnnConfig = transducer.qnnConfig

            if (qnnConfig.backendLib.isEmpty()) {
                throw IllegalArgumentException("You should provide libQnnHtp.so for qnn")
            }

            config.modelConfig.tokens =
                copyAssetToInternalStorage(config.modelConfig.tokens, this)

            if (transducer.encoder.isNotEmpty()) {
                transducer.encoder =
                    copyAssetToInternalStorage(transducer.encoder, this)
            }

            if (transducer.decoder.isNotEmpty()) {
                transducer.decoder =
                    copyAssetToInternalStorage(transducer.decoder, this)
            }

            if (transducer.joiner.isNotEmpty()) {
                transducer.joiner =
                    copyAssetToInternalStorage(transducer.joiner, this)
            }

            if (qnnConfig.contextBinary.isNotEmpty()) {
                qnnConfig.contextBinary =
                    copyAssetListToInternalStorage(qnnConfig.contextBinary, this)
            }

            if (config.hr.lexicon.isNotEmpty()) {
                config.hr.lexicon = copyAssetToInternalStorage(config.hr.lexicon, this)
            }

            if (config.hr.ruleFsts.isNotEmpty()) {
                config.hr.ruleFsts = copyAssetToInternalStorage(config.hr.ruleFsts, this)
            }

            assetManager = null
        }

        recognizer = OnlineRecognizer(
            assetManager = assetManager,
            config = config,
        )
    }
}
