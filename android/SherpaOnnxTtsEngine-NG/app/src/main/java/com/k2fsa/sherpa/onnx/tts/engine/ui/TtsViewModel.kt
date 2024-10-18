package com.k2fsa.sherpa.onnx.tts.engine.ui

import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeech.OnInitListener
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.lifecycle.ViewModel
import com.k2fsa.sherpa.onnx.tts.engine.App
import java.util.Locale

class TtsViewModel : ViewModel() {

    // https://developer.android.com/reference/kotlin/android/speech/tts/TextToSpeech.OnInitListener
    private val onInitListener = OnInitListener { status ->
        when (status) {
            TextToSpeech.SUCCESS -> Log.i(TAG, "Init tts succeded")
            TextToSpeech.ERROR -> Log.i(TAG, "Init tts failed")
            else -> Log.i(TAG, "Unknown status $status")
        }
    }

    // https://developer.android.com/reference/kotlin/android/speech/tts/UtteranceProgressListener
    private val utteranceProgressListener = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {
            Log.i(TAG, "onStart: $utteranceId")
        }

        override fun onStop(utteranceId: String?, interrupted: Boolean) {
            Log.i(TAG, "onStop: $utteranceId, $interrupted")
            super.onStop(utteranceId, interrupted)
        }

        override fun onError(utteranceId: String?, errorCode: Int) {
            Log.i(TAG, "onError: $utteranceId, $errorCode")
            super.onError(utteranceId, errorCode)
        }

        override fun onDone(utteranceId: String?) {
            Log.i(TAG, "onDone: $utteranceId")
        }

        @Deprecated("Deprecated in Java")
        override fun onError(utteranceId: String?) {
            Log.i(TAG, "onError: $utteranceId")
        }
    }

    val tts = TextToSpeech(App.instance, onInitListener, "com.k2fsa.sherpa.onnx.tts.engine")

    init {
//        tts.setLanguage(Locale(TtsEngine.lang!!))
        tts.setOnUtteranceProgressListener(utteranceProgressListener)
    }

    override fun onCleared() {
        super.onCleared()
        tts.shutdown()
    }
}