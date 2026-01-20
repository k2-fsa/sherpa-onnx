package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.res.stringResource

@Composable
fun TextFieldDialog(
    title: String,
    initialText: String,
    onDismissRequest: () -> Unit,
    onConfirm: (String) -> Unit
) {
    var textValue by rememberSaveable { mutableStateOf(initialText) }
    AlertDialog(onDismissRequest = onDismissRequest,
        title = {
            Text(title)
        },
        text = {
            OutlinedTextField(
                value = textValue, onValueChange = { textValue = it },
            )
        },
        dismissButton = {
            TextButton(onClick = onDismissRequest) {
                Text(stringResource(id = android.R.string.cancel))
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onConfirm(textValue)
            }) {
                Text(stringResource(id = android.R.string.ok))
            }
        }
    )
}