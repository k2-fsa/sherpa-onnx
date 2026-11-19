package com.k2fsa.sherpa.onnx.tts.engine.synthesizer

import android.util.Log
import com.k2fsa.sherpa.onnx.tts.engine.conf.TtsConfig
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.DelayQueue
import java.util.concurrent.Delayed
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min


interface ImplCache {
    fun destroy()
    fun canDestroy(): Boolean
}


internal class SynthesizerCache {
    companion object {
        const val TAG = "SynthesizerCache"

        private val delayTime: Int
            get() = 1000 * 60 * min(1, TtsConfig.timeoutDestruction.value)

        private val maxCacheSize: Int
            get() = max(1, TtsConfig.cacheSize.value)
    }

    private val delayQueue = DelayQueue<DelayedDestroyTask>()
    private val queueMap = ConcurrentHashMap<String, DelayedDestroyTask>()
    private val executor = Executors.newSingleThreadExecutor()

    private var isTaskRunning = false
    private fun ensureTaskRunning() {
        if (isTaskRunning) return

        synchronized(delayQueue) {
            isTaskRunning = true
            executor.execute {
                while (true) {
                    if (delayQueue.isEmpty()) {
                        break
                    } else {
                        val task = delayQueue.take()
                        if (task.obj.canDestroy()) {
                            Log.d(TAG, "ensureTaskRunning: ${task.id} is destroyable.")
                            task.obj.destroy()
                            queueMap.remove(task.id)
                        } else {
                            Log.d(TAG, "ensureTaskRunning: ${task.id} is running, not destroyable.")
                            delayQueue.add(task.apply { reset() })
                        }
                    }
                }
                isTaskRunning = false
            }
        }
    }

    private fun limitSize() {
        if (queueMap.size > maxCacheSize) {
            val oldestEntry =
                queueMap.entries.minByOrNull { it.value.getDelay(TimeUnit.MILLISECONDS) }
            oldestEntry?.let {
                if (it.value.obj.canDestroy()) {
                    it.value.obj.destroy()
                    queueMap.remove(it.key)
                    delayQueue.remove(it.value)
                }
            }
        }
    }

    @Synchronized
    fun cache(id: String, obj: ImplCache) {
        limitSize()

        val task = DelayedDestroyTask(delayTime = delayTime, id, obj)
        delayQueue.add(task)
        queueMap[id] = task
        ensureTaskRunning()
    }

    @Synchronized
    fun getById(id: String): ImplCache? {
        limitSize()

        queueMap[id]?.let {
            if (it.getDelay(TimeUnit.MILLISECONDS) <= 1000 * 10) { // 小于10s便重置
                it.reset()
            }
            return it.obj
        }
        return null
    }

    class DelayedDestroyTask(private val delayTime: Int, val id: String, val obj: ImplCache) :
        Delayed {
        private var expireTime: Long = 0L

        init {
            reset()
        }

        fun reset() {
            expireTime = System.currentTimeMillis() + delayTime
        }

        override fun compareTo(other: Delayed?): Int =
            getDelay(TimeUnit.MILLISECONDS).compareTo(
                (other as DelayedDestroyTask).getDelay(TimeUnit.MILLISECONDS)
            )


        override fun getDelay(unit: TimeUnit?): Long =
            unit?.convert(expireTime - System.currentTimeMillis(), TimeUnit.MILLISECONDS) ?: 0
    }
}
