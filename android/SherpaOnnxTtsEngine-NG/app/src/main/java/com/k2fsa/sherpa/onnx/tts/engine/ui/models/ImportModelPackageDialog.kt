package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.documentfile.provider.DocumentFile
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.service.ModelManagerService
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ModelPackageManager
import com.k2fsa.sherpa.onnx.tts.engine.utils.grantReadPermission
import com.k2fsa.sherpa.onnx.tts.engine.utils.longToast

@Composable
fun ImportModelPackageDialog(onDismissRequest: () -> Unit) {
    var showTipsDialog by remember { mutableStateOf(false) }

    if (showTipsDialog)
        AlertDialog(onDismissRequest = onDismissRequest,
            title = { Text(stringResource(id = R.string.import_model_package)) },
            text = { Text(stringResource(R.string.task_added_tips)) },
            confirmButton = {
                TextButton(onClick = onDismissRequest) {
                    Text(stringResource(id = android.R.string.ok))
                }
            }
        )

    val context = LocalContext.current
    val filepicker =
        rememberLauncherForActivityResult(contract = ActivityResultContracts.OpenDocument()) { uri ->
            if (uri == null) {
                onDismissRequest()
                return@rememberLauncherForActivityResult
            }

            try {
                uri.grantReadPermission(context.contentResolver)
            } catch (e: Exception) {
                context.longToast(R.string.unable_grant_read_permission)
                onDismissRequest()
                return@rememberLauncherForActivityResult
            }

            val file = DocumentFile.fromSingleUri(context, uri)?.name
            if (file == null) {
                context.longToast(R.string.unable_get_file_name)
                onDismissRequest()
                return@rememberLauncherForActivityResult
            }

            if (ModelPackageManager.supportedTypes.none { file.endsWith(it) }) {
                context.longToast(
                    R.string.only_ext_files_are_supported,
                    ModelPackageManager.supportedTypes.joinToString(" ")
                )

                onDismissRequest()
                return@rememberLauncherForActivityResult
            }

            showTipsDialog = true

            context.startService(Intent(context, ModelManagerService::class.java).apply {
                data = uri
            })
        }

    LaunchedEffect(key1 = Unit) {
        filepicker.launch(arrayOf("*/*"))
    }


}