package com.k2fsa.sherpa.onnx.tts.engine.service

import android.content.Intent
import android.util.Log
import com.k2fsa.sherpa.onnx.tts.engine.NotificationConst
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ModelPackageManager
import com.k2fsa.sherpa.onnx.tts.engine.utils.NotificationUtils.sendNotification
import com.k2fsa.sherpa.onnx.tts.engine.utils.grantReadWritePermission
import kotlinx.coroutines.launch
import java.io.OutputStream

class ModelExportService : IProgressTaskService("model_export", R.string.export) {
    companion object {
        const val TAG = "ModelExportService"

        const val EXTRA_MODELS = "models"
        const val EXTRA_TYPE = "type"
    }

    override fun onTimedOut() {
        sendNotification(
            channelId = NotificationConst.MODEL_EXPORT_CHANNEL,
            title = getString(R.string.export_failed),
            content = getString(R.string.timed_out)
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        fun error(msg: String): Int {
            Log.e(TAG, msg)
            stopSelf()
            return super.onStartCommand(intent, flags, startId)
        }

        val uri = intent?.data ?: run {
            return error("No uri to export")
        }

        val models = intent.getStringArrayExtra(EXTRA_MODELS) ?: run {
            return error("No models to export")
        }

        val type = intent.getStringExtra(EXTRA_TYPE) ?: run {
            return error("No type to export")
        }

        runCatching {
            uri.grantReadWritePermission(contentResolver)
        }.onFailure {
            return error("Failed to grant uri permission: $uri")
        }

        updateNotification(getString(R.string.export), 0, "", models.joinToString())

        val out = contentResolver.openOutputStream(uri) ?: run {
            return error("Failed to open output stream")
        }
        mScope.launch {
            out.use {
                execute(type, it, models.toList())

                sendNotification(
                    channelId = NotificationConst.MODEL_EXPORT_CHANNEL,
                    title = getString(R.string.export_finished),
                    content = models.joinToString()
                )
                stopSelf()
            }
        }


        return super.onStartCommand(intent, flags, startId)
    }

    private suspend fun execute(type: String, ous: OutputStream, models: List<String>) {
        ModelPackageManager.exportModelsToZip(
            models = models,
            type = type,
            ous = ous,
            onZipProgress = { name, entrySize, bytes ->
                val progress = (bytes * 100 / entrySize).toInt()
                updateNotification(getString(R.string.zipping), progress, name, "")
            }
        )
    }
}