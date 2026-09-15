package com.k2fsa.sherpa.onnx.tts.engine.service

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.SampleTextManager
import com.k2fsa.sherpa.onnx.tts.engine.utils.toCode
import com.k2fsa.sherpa.onnx.tts.engine.utils.toLocaleFromIso3

class GetSampleText : Activity() {
    private fun getCode(): String {
        val language = intent.getStringExtra("language") ?: ""
        val country = intent.getStringExtra("country") ?: ""
        val variant = intent.getStringExtra("variant") ?: ""

        return when {
            language.isNotEmpty() && country.isNotEmpty() && variant.isNotEmpty() -> {
                "$language-$country-$variant"
            }

            language.isNotEmpty() && country.isNotEmpty() -> {
                "$language-$country"
            }

            language.isNotEmpty() -> {
                language
            }

            else -> {
                ""
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        var result = TextToSpeech.LANG_AVAILABLE
        var text = ""
        val code = getCode()
        println("GetSampleText: code=$code")

        text = SampleTextManager.getSampleText(code).ifEmpty {
            result = TextToSpeech.LANG_NOT_SUPPORTED
            ""
        }

        val intent = Intent().apply {
            if (result == TextToSpeech.LANG_AVAILABLE) {
                putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT, text)
            } else {
                putExtra("sampleText", text)
            }
        }

        println("GetSampleText: result=${result}, text=$text")
        setResult(result, intent)
        finish()
    }
}