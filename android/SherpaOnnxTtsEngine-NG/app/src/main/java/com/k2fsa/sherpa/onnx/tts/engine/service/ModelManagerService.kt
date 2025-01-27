package com.k2fsa.sherpa.onnx.tts.engine.service

import android.content.Intent
import android.util.Log
import androidx.core.net.toUri
import androidx.documentfile.provider.DocumentFile
import com.drake.net.component.Progress
import com.k2fsa.sherpa.onnx.tts.engine.NotificationConst
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.conf.AppConfig
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ModelPackageManager
import com.k2fsa.sherpa.onnx.tts.engine.utils.NotificationUtils.sendNotification
import com.k2fsa.sherpa.onnx.tts.engine.utils.fileName
import com.k2fsa.sherpa.onnx.tts.engine.utils.formatFileSize
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch

class ModelManagerService :
    IProgressTaskService(
        NotificationConst.MODEL_PACKAGE_INSTALLER_CHANNEL,
        R.string.model_package_installer
    ) {
    companion object {
        const val TAG = "ModelManagerService"

        const val EXTRA_FILE_NAME = "file_name"
    }

    // [1MB / 25MB] 100 KB/s
    private fun Progress.toNotificationContent(): String =
        "[${currentSize()} / ${totalSize()}] \t ${speedSize()}/s"

    override fun onTimedOut() {
        sendNotification(
            channelId = NotificationConst.MODEL_PACKAGE_INSTALLER_CHANNEL,
            title = getString(R.string.model_install_failed),
            content = getString(R.string.timed_out)
        )
    }

    override fun onStartCommand(intent: Intent, flags: Int, startId: Int): Int {
        val mUri = intent.data?.toString() ?: run {
            Log.e(TAG, "onStartCommand: uri is null")
            stopSelf()
            return super.onStartCommand(intent, flags, startId)
        }
        val mFileName = intent.getStringExtra(EXTRA_FILE_NAME) ?: kotlin.run {
            Log.d(TAG, "onStartCommand: fileName is null, from local file install")
            ""
        }
        Log.i(TAG, "onStartCommand: uri=$mUri, fileName=$mFileName")

        updateNotification(
            "",
            0,
            getString(if (mFileName.isEmpty()) R.string.model_package_installer else R.string.downloading),
            mFileName
        )

        mScope.launch {
            runCatching {
                execute(mUri, mFileName)
            }.onFailure {
                if (it is CancellationException) return@onFailure

                Log.e(TAG, "onStartCommand: execute failed", it)
                sendNotification(
                    channelId = NotificationConst.MODEL_PACKAGE_INSTALLER_CHANNEL,
                    title = getString(R.string.model_install_failed),
                    content = it.message ?: getString(R.string.error)
                )
            }
            stopSelf()
        }

        return super.onStartCommand(intent, flags, startId)
    }

    private suspend fun execute(uriString: String, fileName: String) {
        Log.d(TAG, "execute: $uriString, $fileName")

        fun updateUnzipProgress(file: String, total: Long, current: Long) {
            val name = file.substringAfterLast('/')
            val str =
                "${current.formatFileSize(this)} / ${total.formatFileSize(this)}"
            updateNotification(
                getString(R.string.unzipping),
                progress = ((current / total.toDouble()) * 100).toInt(),
                title = name,
                content = str,
            )
        }

        fun updateStartMoveFiles() {
            updateNotification(
                "",
                progress = -1,
                title = getString(R.string.moving_files),
                content = "..."
            )
        }

        val ok = if (fileName.isBlank()) {
            val uri = uriString.toUri()
            val file = DocumentFile.fromSingleUri(this, uri)
            val name = file?.name ?: throw IllegalArgumentException("file is null: uri=${uri}")
            val type = name.substringAfter(".")

            val ins = contentResolver.openInputStream(uri)
                ?: throw IllegalArgumentException("openInputStream return null: uri=${uri}")
            Log.d(TAG, "execute: type=$type")
            ModelPackageManager.installPackage(
                type,
                ins,
                onUnzipProgress = ::updateUnzipProgress,
                onStartMoveFiles = ::updateStartMoveFiles
            )
        } else {
            val url = if (AppConfig.ghProxyUrl.value.isEmpty())
                uriString
            else
                "${AppConfig.ghProxyUrl.value.removeSuffix("/")}/$uriString"
            ModelPackageManager.installPackageFromUrl(
                url = url,
                fileName = fileName,
                onDownloadProgress = {
                    updateNotification(
                        summary = getString(R.string.downloading),
                        progress = it.progress(),
                        title = fileName,
                        content = it.toNotificationContent()
                    )
                },
                onUnzipProgress = ::updateUnzipProgress,
                onStartMoveFiles = ::updateStartMoveFiles
            )
        }

        sendNotification(
            channelId = NotificationConst.MODEL_PACKAGE_INSTALLER_CHANNEL,
            title = getString(if (ok) R.string.model_installed else R.string.model_install_failed),
            content = fileName.ifBlank {
                try {
                    uriString.toUri().fileName(this)
                } catch (e: Exception) {
                    Log.e(TAG, "execute: unable to get file name from uri=$uriString", e)
                    ""
                }
            }
        )
    }

}