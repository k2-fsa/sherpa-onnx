package com.k2fsa.sherpa.onnx.tts.engine.ui

import android.Manifest
import android.os.Build
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.platform.LocalContext
import com.google.accompanist.permissions.ExperimentalPermissionsApi
import com.google.accompanist.permissions.PermissionStatus
import com.google.accompanist.permissions.rememberPermissionState

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun NotificationPermissionChecker() {
    val context = LocalContext.current
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // A13
        val notificationPermission =
            rememberPermissionState(permission = Manifest.permission.POST_NOTIFICATIONS)

        LaunchedEffect(key1 = notificationPermission) {
            when (notificationPermission.status) {
                is PermissionStatus.Denied -> {
//                    context.longToast(context.getString(R.string.notification_goto_settings_enable))
                    notificationPermission.launchPermissionRequest()
                }

                is PermissionStatus.Granted -> {
                }
            }
        }
    }
}