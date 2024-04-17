package com.k2fsa.sherpa.onnx.slid

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.SpokenLanguageIdentification
import com.k2fsa.sherpa.onnx.getSpokenLanguageIdentificationConfig
import java.util.Locale


object Slid {
    private var _slid: SpokenLanguageIdentification? = null

    private var _localeMap = mutableMapOf<String, String>()
    val slid: SpokenLanguageIdentification
        get() {
            return _slid!!
        }
    val localeMap : Map<String, String>
            get() {
                return _localeMap
            }

    fun initSlid(assetManager: AssetManager? = null, numThreads: Int = 1) {
        synchronized(this) {
            if (_slid == null) {

                Log.i(TAG, "Initializing slid")
                val config =
                    getSpokenLanguageIdentificationConfig(type = 0, numThreads = numThreads)!!
                _slid = SpokenLanguageIdentification(assetManager, config)
            }

            if (_localeMap.isEmpty()) {
                val allLang =  Locale.getISOLanguages();
                for (lang in allLang) {
                    val locale = Locale(lang)
                    _localeMap[lang] = locale.displayName
                }
            }
        }
    }
}