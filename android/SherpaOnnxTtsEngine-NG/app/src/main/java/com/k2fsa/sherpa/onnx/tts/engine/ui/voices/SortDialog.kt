package com.k2fsa.sherpa.onnx.tts.engine.ui.voices

import androidx.compose.runtime.Composable
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.CancelButton
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.OkButton

@Composable
fun SortDialog(onDismissRequest: () -> Unit, onConfirm: () -> Unit) {
    AppDialog(onDismissRequest = onDismissRequest, title = {


    }, content = {

    }, buttons = {
        CancelButton {
            onDismissRequest()
        }

        OkButton {
            onConfirm()
        }
    }
    )
}