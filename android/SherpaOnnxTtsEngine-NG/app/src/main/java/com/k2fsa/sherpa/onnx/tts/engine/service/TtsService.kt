package com.k2fsa.sherpa.onnx.tts.engine.service

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.speech.tts.Voice
import android.util.Log
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.conf.TtsConfig
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager.toOfflineTtsConfig
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigVoiceManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.SynthesizerManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import com.k2fsa.sherpa.onnx.tts.engine.ui.TAG
import com.k2fsa.sherpa.onnx.tts.engine.utils.longToast
import com.k2fsa.sherpa.onnx.tts.engine.utils.newLocaleFromCode
import com.k2fsa.sherpa.onnx.tts.engine.utils.toByteArray
import java.util.Locale
import kotlin.math.min

/*
https://developer.android.com/reference/java/util/Locale#getISO3Language()
https://developer.android.com/reference/java/util/Locale#getISO3Country()

eng, USA,
eng, USA, POSIX
eng,
eng, GBR
afr,
afr, NAM
afr, ZAF
agq
agq, CMR
aka,
aka, GHA
amh,
amh, ETH
ara,
ara, 001
ara, ARE
ara, BHR,
deu
deu, AUT
deu, BEL
deu, CHE
deu, ITA
deu, ITA
deu, LIE
deu, LUX
spa,
spa, 419
spa, ARG,
spa, BRA
fra,
fra, BEL,
fra, FRA,

E  Failed to check TTS data, no activity found for Intent
{ act=android.speech.tts.engine.CHECK_TTS_DATA pkg=com.k2fsa.sherpa.chapter5 })

E Failed to get default language from engine com.k2fsa.sherpa.chapter5
Engine failed voice data integrity check (null return)com.k2fsa.sherpa.chapter5
Failed to get default language from engine com.k2fsa.sherpa.chapter5

*/

class TtsService : TextToSpeechService() {
    companion object {
        const val NOT_SET_VOICE_NAME = "NOT_SET_VOICE"
    }

    private var languages: List<Locale> = emptyList()
    override fun onCreate() {
        Log.i(TAG, "onCreate tts service")

        ConfigModelManager.load()
        languages = ConfigModelManager.languages().map { newLocaleFromCode(it) }

        super.onCreate()
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy tts service")
        super.onDestroy()
    }

    // https://developer.android.com/reference/kotlin/android/speech/tts/TextToSpeechService#onislanguageavailable
    override fun onIsLanguageAvailable(_lang: String?, _country: String?, _variant: String?): Int {
        Log.d(TAG, "onIsLanguageAvailable: $_lang, $_country $_variant")
        val lang = _lang ?: ""
        val country = _country ?: ""

        languages.forEach {
            val l = it.isO3Language
            val c = it.isO3Country

            if (l == lang && c == country) {
                return TextToSpeech.LANG_COUNTRY_AVAILABLE
            } else if (l == lang) {
                return TextToSpeech.LANG_AVAILABLE
            }
        }

        return TextToSpeech.LANG_NOT_SUPPORTED
    }

    // https://developer.android.com/reference/kotlin/android/speech/tts/TextToSpeechService#onLoadLanguage(kotlin.String,%20kotlin.String,%20kotlin.String)
    override fun onLoadLanguage(_lang: String?, _country: String?, _variant: String?): Int {
        return onIsLanguageAvailable(_lang, _country, _variant)
    }

    override fun onGetLanguage(): Array<String> {
        return arrayOf("", "", "")
    }

    override fun onLoadVoice(voiceName: String?): Int {
        Log.i(TAG, "onLoadVoice: $voiceName")
        return onIsValidVoiceName(voiceName)
    }

    override fun onGetVoices(): MutableList<Voice> {
        val list = mutableListOf<Voice>()
        list.add(
            Voice(
                NOT_SET_VOICE_NAME, Locale("zh", "CN"), Voice.QUALITY_NORMAL,
                /* latency = */ Voice.LATENCY_NORMAL,
                /* requiresNetworkConnection = */ false,
                /* features = */ setOf()
            )
        )
        list.add(
            Voice(
                NOT_SET_VOICE_NAME, Locale("en", "US"), Voice.QUALITY_NORMAL,
                /* latency = */ Voice.LATENCY_NORMAL,
                /* requiresNetworkConnection = */ false,
                /* features = */ setOf()
            )
        )
        ConfigVoiceManager.speakers().forEach {
            val model = ConfigModelManager.models().find { m -> m.id == it.model } ?: return@forEach
            list.add(
                Voice(
                    /* name = */ model.id + "_" + it.id,
                    /* locale = */ newLocaleFromCode(model.lang),
                    /* quality = */ Voice.QUALITY_NORMAL,
                    /* latency = */ Voice.LATENCY_NORMAL,
                    /* requiresNetworkConnection = */ false,
                    /* features = */ setOf(it.name)
                )
            )
        }

        Log.i(TAG, "onGetVoices: ${list.size}")
        return list
    }

    override fun onIsValidVoiceName(voiceName: String?): Int {
        Log.i(TAG, "onIsValidVoiceName: $voiceName")
        if (voiceName.isNullOrBlank())
            return TextToSpeech.ERROR

        return if (voiceName == NOT_SET_VOICE_NAME ||
            com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Voice.from(voiceName).run {
                ConfigVoiceManager.speakers().any { it.id == id } && ConfigModelManager.models()
                    .any { it.id == model }
            }
        ) {
            TextToSpeech.SUCCESS
        } else {
            TextToSpeech.ERROR
        }
    }

    override fun onGetDefaultVoiceNameFor(
        lang: String?,
        country: String?,
        variant: String?
    ): String {
        Log.i(TAG, "onGetDefaultVoiceNameFor: $lang, $country, $variant")
        return NOT_SET_VOICE_NAME
    }

    override fun onStop() {}

    private fun getTtsModel(voiceName: String?): Model? {
        Log.d(TAG, "getTtsModel: $voiceName")
        return ConfigModelManager.models()
            .run { if (voiceName == null || voiceName == NOT_SET_VOICE_NAME) null else this }
            ?.find { it.id == voiceName }
            ?: ConfigModelManager.models().find { it.id == TtsConfig.voice.value.model }
            ?: ConfigModelManager.models().getOrNull(0)
    }

    override fun onSynthesizeText(request: SynthesisRequest?, callback: SynthesisCallback?) {
        if (request == null || callback == null) {
            return
        }
        val language = request.language
        val country = request.country
        val variant = request.variant
        val text = request.charSequenceText.toString()
        val voiceName: String? = request.voiceName

        val ret = onIsLanguageAvailable(language, country, variant)
        if (ret == TextToSpeech.LANG_NOT_SUPPORTED) {
            callback.error()
            return
        }
        Log.i(TAG, "text: $text")

        val voice =
            if (voiceName.isNullOrBlank() || voiceName == NOT_SET_VOICE_NAME) TtsConfig.voice.value else
                com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Voice.from(voiceName)
        val ttsModel = ConfigModelManager.models().find { it.id == voice.model }
        Log.i(TAG, "ttsConfig: $ttsModel")
        if (ttsModel == null) {
            Log.e(TAG, "tts not found")
            longToast(R.string.tts_config_not_set)
            callback.error()
            return
        }

        val tts = SynthesizerManager.getTTS(ttsModel.toOfflineTtsConfig())

        // Note that AudioFormat.ENCODING_PCM_FLOAT requires API level >= 24
        // callback.start(tts.sampleRate(), AudioFormat.ENCODING_PCM_FLOAT, 1)

        callback.start(tts.sampleRate(), AudioFormat.ENCODING_PCM_16BIT, 1)

        if (text.isBlank() || text.isEmpty()) {
            callback.done()
            return
        }

        val ttsCallback = { floatSamples: FloatArray ->
            // convert FloatArray to ByteArray
            val samples = floatSamples.toByteArray()
            val maxBufferSize: Int = callback.maxBufferSize
            var offset = 0
            while (offset < samples.size) {
                val bytesToWrite = min(maxBufferSize, samples.size - offset)
                callback.audioAvailable(samples, offset, bytesToWrite)
                offset += bytesToWrite
            }

        }

        val speed = request.speechRate / 100f
        Log.i(
            TAG,
            "voice: ${voice}, text: $text, speechRate: ${request.speechRate} (speed: ${speed})"
        )
        tts.generateWithCallback(
            text = text,
            sid = voice.id,
            speed = speed,
            callback = ttsCallback,
        )

        callback.done()
    }
}