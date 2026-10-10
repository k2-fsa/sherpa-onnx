package com.k2fsa.sherpa.onnx.tts.engine.utils

import kotlinx.coroutines.*

@OptIn(DelicateCoroutinesApi::class)
class ThrottleUtil(private val scope: CoroutineScope = GlobalScope, val time: Long = 100L) {
    var job: Job? = null

    fun runAction(
        dispatcher: CoroutineDispatcher = Dispatchers.Main,
        action: suspend () -> Unit,
    ) {
        job?.cancel()
        job = null
        job = scope.launch(dispatcher) {
            delay(time)
            action.invoke()
        }
    }
}