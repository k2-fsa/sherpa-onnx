package com.k2fsa.sherpa.onnx.tts.engine

import android.app.Activity
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.speech.tts.TextToSpeech

class GetSampleText : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        var result = TextToSpeech.LANG_AVAILABLE
        var text: String = ""
        when(TtsEngine.lang) {
            "eng" -> {
                text = "This is a text-to-speech engine with next generation Kaldi"
            }
            "zho", "cmn" -> {
                text = "使用新一代 Kaldi 进行语音合成"
            }
            else -> {
                result = TextToSpeech.LANG_NOT_SUPPORTED
            }
        }

        val intent = Intent().apply{
            if(result == TextToSpeech.LANG_AVAILABLE) {
                putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT, text)
            } else {
                putExtra("sampleText", text)
            }
        }

        setResult(result, intent)
        finish()
    }
}