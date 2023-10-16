// add by longsm at 2023/10/13
package com.k2fsa.sherpa.onnx

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.text.TextUtils
import android.text.method.ScrollingMovementMethod
import android.util.Log
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import org.java_websocket.handshake.ServerHandshake
import java.net.URI
import java.net.URISyntaxException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

private const val TAG = "sherpa-onnx"
private const val REQUEST_RECORD_AUDIO_PERMISSION = 200

class MainActivity : AppCompatActivity(), MyWebsocketClient.WebsocketClientCallback {
    private val permissions: Array<String> = arrayOf(Manifest.permission.RECORD_AUDIO)

    private var audioRecord: AudioRecord? = null
    private lateinit var recordButton: Button
    private lateinit var connectButton: Button
    private lateinit var textView: TextView
    private lateinit var etUrl: EditText
    private var recordingThread: Thread? = null

    private var websocketClient: MyWebsocketClient? = null

    private val audioSource = MediaRecorder.AudioSource.MIC
    private val sampleRateInHz = 16000
    private val channelConfig = AudioFormat.CHANNEL_IN_MONO

    // Note: We don't use AudioFormat.ENCODING_PCM_FLOAT
    // since the AudioRecord.read(float[]) needs API level >= 23
    // but we are targeting API level >= 21
    private val audioFormat = AudioFormat.ENCODING_PCM_16BIT
    private var idx: Long = 0
    private var lastText: String = ""

    @Volatile
    private var isRecording: Boolean = false

    @Volatile
    private var isConnected: Boolean = false

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

        recordButton = findViewById(R.id.record_button)
        recordButton.setOnClickListener { onclick() }

        connectButton = findViewById(R.id.connect_button)
        connectButton.setOnClickListener { onclickConnect() }

        textView = findViewById(R.id.my_text)
        textView.movementMethod = ScrollingMovementMethod()

        recordButton.isEnabled = false

        etUrl = findViewById(R.id.et_uri)
    }

    private fun onclickConnect() {
        if (!isConnected) {
            val etUrlStr = etUrl.text.toString().trim()
            var uriStr = "ws://172.28.13.167:6006"
            if (!TextUtils.isEmpty(etUrlStr)) {
                uriStr = etUrlStr
            }
            try {
                val uri = URI(uriStr)
                websocketClient = MyWebsocketClient(uri)
                websocketClient?.setClientCallback(this)
                websocketClient?.connect()
            } catch (e: URISyntaxException) {
                Log.e(TAG, "URISyntaxException === >> $e")
            }
        } else {
            Log.e(TAG, "onclick disconnect")
            websocketClient?.close()
            websocketClient = null
        }

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
            connectButton.isEnabled = false
            Log.i(TAG, "Started recording")
        } else {
            isRecording = false
            audioRecord!!.stop()
            audioRecord!!.release()
            audioRecord = null
            recordButton.setText(R.string.start)
            connectButton.isEnabled = true
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

                val buffer = ByteBuffer.allocate(4 * samples.size)
                    .order(ByteOrder.LITTLE_ENDIAN) // float is sizeof 4. allocate enough buffer


                for (f in samples) {
                    buffer.putFloat(f)
                }
                buffer.rewind()
                buffer.flip()
                buffer.order(ByteOrder.LITTLE_ENDIAN)

                if (isConnected) {
                    websocketClient?.send(buffer.array()) // send buf to server
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

    override fun onOpen(handshakedata: ServerHandshake?) {
        Log.i(TAG, "onOpen === >>")
        isConnected = true
        runOnUiThread {
            recordButton.isEnabled = true
            connectButton.text = getString(R.string.disconnect)
        }
    }

    private val gson = Gson()
    private val recognitionText = hashMapOf<Long, String>()

    private fun getDisplayResult(): String {
        var i = 0
        var ans = ""
        for ((key,value) in recognitionText){
            if (value == ""){
                continue
            }
            ans += " $i : ${recognitionText[key]}\n"
            i += 1
        }
        return ans

    }

    override fun onMessage(message: String?) {
        Log.i(TAG, "onMessage === >> $message")
        val speechContent = gson.fromJson<SpeechContent>(
            message,
            object : TypeToken<SpeechContent?>() {}.type
        )

        val text = speechContent.text
        val segment = speechContent.segment
        Log.i(TAG, "text === >> $text")

        recognitionText[segment] = text
        runOnUiThread {
            textView.text = getDisplayResult()
        }
    }

    override fun onClose(code: Int, reason: String?, remote: Boolean?) {
        Log.i(TAG, "onClose === >> code$code reason$reason remote$remote")
        isConnected = false
        runOnUiThread {
            recordButton.isEnabled = false
            connectButton.text = getString(R.string.connect)
            textView.text = getString(R.string.hint)
        }

    }

    override fun onError(ex: Exception?) {
        Log.i(TAG, "onError === >> $ex")
        runOnUiThread {
            textView.text = "onError === >> $ex"
        }

    }
}
