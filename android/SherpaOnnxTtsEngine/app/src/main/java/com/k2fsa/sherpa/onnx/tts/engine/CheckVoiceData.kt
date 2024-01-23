package com.k2fsa.sherpa.onnx.tts.engine

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.speech.tts.TextToSpeech

class CheckVoiceData : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val intent = Intent().apply {
            putStringArrayListExtra(TextToSpeech.Engine.EXTRA_AVAILABLE_VOICES, arrayListOf(TtsEngine.lang))
            putStringArrayListExtra(TextToSpeech.Engine.EXTRA_UNAVAILABLE_VOICES, arrayListOf())
        }
        setResult(TextToSpeech.Engine.CHECK_VOICE_DATA_PASS, intent)
        finish()
    }
}