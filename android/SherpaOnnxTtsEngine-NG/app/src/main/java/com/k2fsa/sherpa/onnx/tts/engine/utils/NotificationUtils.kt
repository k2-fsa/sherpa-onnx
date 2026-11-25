package com.k2fsa.sherpa.onnx.tts.engine.utils

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.annotation.RequiresApi
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.app
import com.k2fsa.sherpa.onnx.tts.engine.ui.MainActivity
import java.util.concurrent.atomic.AtomicLong


val pendingIntentFlags =
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
        PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_MUTABLE or
                PendingIntent.FLAG_ALLOW_UNSAFE_IMPLICIT_INTENT
    } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
    } else {
        PendingIntent.FLAG_UPDATE_CURRENT
    }

val notificationManager
    get() = app.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

@Suppress("DEPRECATION")
object NotificationUtils {
    const val UNSPECIFIED_ID = -1

    private val mAtomLong = AtomicLong(0)

    @Synchronized
    fun nextNotificationId(): Int = mAtomLong.incrementAndGet().toInt()

    fun Context.notificationBuilder(channelId: String): Notification.Builder =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, channelId)
        } else
            Notification.Builder(this)

    @RequiresApi(Build.VERSION_CODES.O)
    fun createChannel(
        id: String, name: String, importance: Int = NotificationManager.IMPORTANCE_HIGH
    ) {
        val chan = NotificationChannel(id, name, importance)
        chan.lightColor = android.graphics.Color.CYAN
        chan.lockscreenVisibility = Notification.VISIBILITY_PUBLIC
        notificationManager.createNotificationChannel(chan)
    }

    fun Context.sendNotification(
        notificationId: Int = nextNotificationId(),
        channelId: String,
        title: String,
        content: String = "",
    ) {
        notificationManager.notify(
            notificationId,
            notificationBuilder(channelId).apply {
                setContentTitle(title)
                setContentText(content)
                setSmallIcon(R.mipmap.ic_launcher)
                setAutoCancel(true)
                setContentIntent(
                    PendingIntent.getActivity(
                        this@sendNotification,
                        0,
                        Intent(this@sendNotification, MainActivity::class.java),
                        pendingIntentFlags
                    )
                )
            }.build()
        )
    }
}