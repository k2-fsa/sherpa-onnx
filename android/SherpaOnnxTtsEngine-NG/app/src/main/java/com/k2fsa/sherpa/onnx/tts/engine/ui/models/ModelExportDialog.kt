package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import android.content.Intent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.service.ModelExportService
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.CancelButton
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.OkButton
import com.k2fsa.sherpa.onnx.tts.engine.utils.grantReadWritePermission

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelExportDialog(show: Boolean, onDismissRequest: () -> Unit, models: List<Model>) {
    var showTips by remember { mutableStateOf(false) }
    if (showTips)
        TaskAddedTipsDialog(
            onDismissRequest = {
                showTips = false
            },
            title = stringResource(id = R.string.export)
        )

    val context = LocalContext.current
    var currentType by rememberSaveable { mutableStateOf<String>("zip") }
    val filepicker =
        rememberLauncherForActivityResult(contract = ActivityResultContracts.CreateDocument("*/*")) {
            it?.let { uri ->
                uri.grantReadWritePermission(context.contentResolver)

                context.startService(Intent(context, ModelExportService::class.java).apply {
                    putExtra(ModelExportService.EXTRA_MODELS, models.map { it.id }.toTypedArray())
                    putExtra(ModelExportService.EXTRA_TYPE, currentType)
                    data = uri
                })

                showTips = true
            }

            onDismissRequest()
        }


    val typeList = remember { listOf("zip", "tar.gz", "tar.xz", "tar.bz2") }
    if (show)
        AppDialog(onDismissRequest = onDismissRequest, title = {
            Text(text = stringResource(id = R.string.export))
        }, content = {
            FlowRow(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
            ) {
                typeList.forEach {
                    val selected = it == currentType
                    FilterChip(
                        selected,
                        modifier = Modifier.padding(horizontal = 4.dp),
                        onClick = { currentType = it },
                        label = {
                            Text(
                                it,
                                fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal
                            )
                        }
                    )
                }
            }
        }, buttons = {
            CancelButton { onDismissRequest() }
            OkButton {
                filepicker.launch("onnx-models.${currentType}")
            }
        })
}