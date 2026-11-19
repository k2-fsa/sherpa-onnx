package com.k2fsa.sherpa.onnx.tts.engine.conf

import com.funny.data_saver.core.DataSaverPreferences
import com.funny.data_saver.core.mutableDataSaverStateOf
import com.k2fsa.sherpa.onnx.tts.engine.App

object AppConfig {
    private val dataSaverPref = DataSaverPreferences(App.instance.getSharedPreferences("app", 0))

    val ghProxyUrl = mutableDataSaverStateOf(dataSaverPref, "ghProxyUrl", "")
}