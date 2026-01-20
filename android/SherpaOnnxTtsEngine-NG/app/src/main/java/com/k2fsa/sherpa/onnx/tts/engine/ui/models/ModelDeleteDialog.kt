package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.github.jing332.tts_server_android.compose.widgets.TextCheckBox
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.CancelButton
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.OkButton

@Composable
internal fun ModelDeleteDialog(
    onDismissRequest: () -> Unit,
    message: String,
    onConfirm: (Boolean) -> Unit,
) {
    var isDeleteFile by rememberSaveable { mutableStateOf(false) }

    AppDialog(onDismissRequest = onDismissRequest,
        title = {
            Text(text = stringResource(R.string.delete_model))
        }, content = {
            Column(
                Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(message, style = MaterialTheme.typography.bodyMedium)
                val color = if (isDeleteFile) MaterialTheme.colorScheme.error else Color.Unspecified
                TextCheckBox(
                    modifier = Modifier.padding(top = 24.dp),
                    colors = CheckboxDefaults.colors(checkedColor = color, uncheckedColor = color),
                    text = { Text(stringResource(R.string.delete_model_file), color = color) },
                    checked = isDeleteFile,
                    onCheckedChange = { isDeleteFile = it })
            }
        }, buttons = {
            Row {
                CancelButton {
                    onDismissRequest()
                }
                OkButton {
                    onConfirm(isDeleteFile)
                }
            }
        })
}