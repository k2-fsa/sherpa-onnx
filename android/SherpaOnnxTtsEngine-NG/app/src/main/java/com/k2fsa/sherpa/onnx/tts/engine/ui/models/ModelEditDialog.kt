package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.CancelButton
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.DenseOutlinedField
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.OkButton

@Composable
fun ModelEditDialog(
    onDismissRequest: () -> Unit,
    model: Model,
    onSave: (Model) -> Unit
) {
    var data by remember { mutableStateOf(model) }
    AppDialog(onDismissRequest = onDismissRequest, title = {
        Text(stringResource(R.string.edit))
    }, content = {
        Content(
            model = data,
            onModelChange = {
                data = it
            }
        )
    }, buttons = {
        Row {
            CancelButton(onClick = onDismissRequest)
            OkButton(Modifier.padding(start = 4.dp), onClick = {
                onSave(data.copy())
                onDismissRequest()
            })
        }
    })
}


@Composable
private fun Content(
    modifier: Modifier = Modifier,
    model: Model,
    onModelChange: (Model) -> Unit
) {
    Column(modifier) {
        DenseOutlinedField(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 4.dp),
            value = model.id,
            onValueChange = {
                onModelChange(model.copy(id = it))
            },
            label = { Text(text = "ID") }
        )

        DenseOutlinedField(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 4.dp),
            value = model.name,
            onValueChange = {
                onModelChange(model.copy(name = it))
            },
            label = { Text(text = stringResource(R.string.display_name)) }
        )

        DenseOutlinedField(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 4.dp),
            value = model.onnx,
            onValueChange = {
                onModelChange(model.copy(onnx = it))
            },
            label = { Text(text = stringResource(R.string.onnx_model_file)) }
        )

        LanguageTextField(
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 4.dp),
            language = model.lang,
            onLanguageChange = { onModelChange(model.copy(lang = it)) }
        )
    }
}