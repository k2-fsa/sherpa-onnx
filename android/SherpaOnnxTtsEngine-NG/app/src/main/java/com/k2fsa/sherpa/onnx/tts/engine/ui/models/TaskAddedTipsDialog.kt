package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.res.stringResource
import com.k2fsa.sherpa.onnx.tts.engine.R

@Composable
fun TaskAddedTipsDialog(onDismissRequest: () -> Unit, title: String) {
    AlertDialog(onDismissRequest = onDismissRequest,
        title = { Text(title) },
        text = { Text(stringResource(R.string.task_added_tips)) },
        confirmButton = {
            TextButton(onClick = onDismissRequest) {
                Text(stringResource(id = android.R.string.ok))
            }
        }
    )
}