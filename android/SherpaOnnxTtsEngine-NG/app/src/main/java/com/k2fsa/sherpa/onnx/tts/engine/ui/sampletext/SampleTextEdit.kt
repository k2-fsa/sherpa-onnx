package com.k2fsa.sherpa.onnx.tts.engine.ui.sampletext

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.DeleteForever
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.ui.theme.SherpaOnnxTtsEngineTheme
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.DeleteForeverIcon
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.DenseOutlinedField

@Preview
@Composable
private fun PreviewSampleTextEdit() {
    SherpaOnnxTtsEngineTheme {
        var list by remember {
            mutableStateOf(listOf("Text1", "Text2"))
        }
        SampleTextEdit(
            modifier = Modifier,
            list = list,
            onListChange = { list = it }
        )
    }
}

@Composable
fun SampleTextEditDialog(
    onDismissRequest: () -> Unit, code: String,
    list: List<String>,
    onConfirm: (List<String>) -> Unit
) {
    var data by remember { mutableStateOf(list) }
    AppDialog(
        onDismissRequest = onDismissRequest, title = { Text(code) },
        content = {
            SampleTextEdit(modifier = Modifier, list = data) {
                data = it
            }
        },
        buttons = {
            Row {
                TextButton(onClick = onDismissRequest) {
                    Text(stringResource(android.R.string.cancel))
                }
                TextButton(onClick = {
                    onConfirm(data)
                    onDismissRequest()
                }) {
                    Text(stringResource(android.R.string.ok))
                }
            }
        }
    )
}

@Composable
fun SampleTextEdit(modifier: Modifier, list: List<String>, onListChange: (List<String>) -> Unit) {
    LazyColumn {
        list.forEachIndexed { index, s ->
            item {
                DenseOutlinedField(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 2.dp, horizontal = 4.dp),
                    value = s,
                    onValueChange = {
                        val newList = list.toMutableList()
                        newList[index] = it
                        onListChange(newList)
                    },
                    label = { Text((index + 1).toString()) },
                    trailingIcon = {
                        IconButton(onClick = {
                            val newList = list.toMutableList()
                            newList.removeAt(index)
                            onListChange(newList)
                        }) {
                           DeleteForeverIcon()
                        }
                    }
                )
            }
        }

        item {
            TextButton(
                modifier = Modifier.fillMaxWidth(),
                onClick = {
                    val newList = list.toMutableList()
                    newList.add("")
                    onListChange(newList)
                }) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Add, contentDescription = null)
                    Text(stringResource(R.string.add))
                }
            }
        }
    }
}