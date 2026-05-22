package com.k2fsa.sherpa.onnx.tts.engine.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import androidx.core.content.ContextCompat
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.ui.MainActivity
import com.k2fsa.sherpa.onnx.tts.engine.utils.NotificationUtils
import com.k2fsa.sherpa.onnx.tts.engine.utils.NotificationUtils.notificationBuilder
import com.k2fsa.sherpa.onnx.tts.engine.utils.ThrottleUtil
import com.k2fsa.sherpa.onnx.tts.engine.utils.pendingIntentFlags
import com.k2fsa.sherpa.onnx.tts.engine.utils.startForegroundCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancel

abstract class IProgressTaskService(
    private val chanId: String, private val chanNameStrId: Int
) :
    Service() {
    companion object {
        const val TAG = "ModelManagerService"

        const val ACTION_NOTIFICATION_CANCEL =
            "com.k2fsa.sherpa.onnx.tts.engine.service.IProgressTaskService.ACTION_NOTIFICATION_CANCEL"

        const val EXTRA_NOTIFICATION_ID = "notification_id"
    }

    override fun onBind(intent: Intent): IBinder? = null

    private var mNotificationId = NotificationUtils.nextNotificationId()
    private val mNotificationReceiver by lazy { NotificationReceiver() }

    inner class NotificationReceiver : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == ACTION_NOTIFICATION_CANCEL) {
                if (mNotificationId == intent.getIntExtra(
                        EXTRA_NOTIFICATION_ID,
                        NotificationUtils.UNSPECIFIED_ID
                    )
                ) {
                    stopSelf()
                }
            }
        }
    }

    override fun onCreate() {
        super.onCreate()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationUtils.createChannel(
                chanId,
                getString(chanNameStrId),
            )
        }

        ContextCompat.registerReceiver(
            this,
            mNotificationReceiver,
            IntentFilter(ACTION_NOTIFICATION_CANCEL),
            ContextCompat.RECEIVER_EXPORTED
        )
    }

    @Suppress("DEPRECATION")
    override fun onDestroy() {
        mNotificationId = NotificationUtils.UNSPECIFIED_ID
        stopForeground(true)
        mScope.cancel()
        unregisterReceiver(mNotificationReceiver)

        super.onDestroy()
    }

    @Suppress("DEPRECATION")
    private fun createNotification(
        summary: String,
        progress: Int,
        title: String,
        content: String
    ): Notification {
        Log.d(TAG, "createNotification: $progress, $title, $content")
        return notificationBuilder(chanId).apply {
            setContentTitle(title)
            setContentText(content)
            setSmallIcon(R.mipmap.ic_launcher)
//            setVisibility(Notification.VISIBILITY_PUBLIC)
            style = Notification.BigTextStyle().setSummaryText(summary).setBigContentTitle(title)
                .bigText(content)

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                setForegroundServiceBehavior(Notification.FOREGROUND_SERVICE_IMMEDIATE)
            }

            setProgress(100, progress, progress == -1)

            val cancelPending = PendingIntent.getBroadcast(
                /* context = */ this@IProgressTaskService,
                /* requestCode = */ 0,
                /* intent = */ Intent(ACTION_NOTIFICATION_CANCEL).apply {
                    putExtra(EXTRA_NOTIFICATION_ID, mNotificationId)
                },
                /* flags = */ pendingIntentFlags
            )
            addAction(
                Notification.Action.Builder(
                    0,
                    getString(android.R.string.cancel),
                    cancelPending
                ).build()
            )
            setContentIntent(
                PendingIntent.getActivity(
                    this@IProgressTaskService,
                    0,
                    Intent(this@IProgressTaskService, MainActivity::class.java),
                    pendingIntentFlags
                )
            )
        }.build()
    }

    abstract fun onTimedOut()

    protected val mScope = CoroutineScope(Dispatchers.IO)

    private val timeoutThrottle = ThrottleUtil(mScope, time = 1000L * 15) //15s
    private fun setTimeout() {
        timeoutThrottle.runAction {
            onTimedOut()
            stopSelf()
        }
    }

    private var mLastUpdateNotification = 0L
    protected fun updateNotification(
        summary: String,
        progress: Int,
        title: String,
        content: String
    ) {
        setTimeout()
        if (mNotificationId != NotificationUtils.UNSPECIFIED_ID) {
            if (SystemClock.elapsedRealtime() - mLastUpdateNotification < 500) return

            Log.d(TAG, "startForegroundCompat: $progress, $title, $content")
            startForegroundCompat(
                mNotificationId,
                createNotification(summary, progress, title, content)
            )
            mLastUpdateNotification = SystemClock.elapsedRealtime()
        }
    }
}