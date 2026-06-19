package com.k2fsa.sherpa.onnx.tts.engine.conf

import com.funny.data_saver.core.DataSaverConverter
import com.funny.data_saver.core.DataSaverPreferences
import com.funny.data_saver.core.mutableDataSaverStateOf
import com.k2fsa.sherpa.onnx.tts.engine.App
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Voice

object TtsConfig {
    private val dataSaverPref = DataSaverPreferences(App.instance.getSharedPreferences("tts", 0))

    init {
        DataSaverConverter.registerTypeConverters<Voice>(
            save = { it.toString() },
            restore = { Voice.from(it) }
        )
    }

    val voice = mutableDataSaverStateOf(
        dataSaverInterface = dataSaverPref,
        key = "voice",
        initialValue = Voice.EMPTY
    )

    val timeoutDestruction = mutableDataSaverStateOf(
        dataSaverInterface = dataSaverPref,
        key = "timeoutDestruction",
        initialValue = 3
    )

    val cacheSize = mutableDataSaverStateOf(
        dataSaverInterface = dataSaverPref,
        key = "cacheSize",
        initialValue = 3
    )

    val threadNum = mutableDataSaverStateOf(
        dataSaverInterface = dataSaverPref,
        key = "threadNum",
        initialValue = 2
    )

}