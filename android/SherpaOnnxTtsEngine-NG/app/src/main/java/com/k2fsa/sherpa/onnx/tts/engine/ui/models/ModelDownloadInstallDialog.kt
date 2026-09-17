package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import android.content.Intent
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import androidx.lifecycle.viewmodel.compose.viewModel
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.service.ModelManagerService
import com.k2fsa.sherpa.onnx.tts.engine.service.ModelManagerService.Companion.EXTRA_FILE_NAME
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.LoadingContent
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.SearchTextFieldInList
import com.k2fsa.sherpa.onnx.tts.engine.utils.clickableRipple
import com.k2fsa.sherpa.onnx.tts.engine.utils.formatFileSize

@Preview
@Composable
private fun PreviewModelDownloadDialog() {
    var show by remember { mutableStateOf(false) }

    ModelDownloadInstallDialog(onDismissRequest = { show = false })
}

@Composable
fun ModelDownloadInstallDialog(
    onDismissRequest: () -> Unit,
) {
    val vm: ModelDownloadInstallViewModel = viewModel()
    val context = LocalContext.current

    LaunchedEffect(key1 = Unit) {
        vm.load()
    }

    var showTips by remember { mutableStateOf(false) }
    if (showTips)
        TaskAddedTipsDialog(
            onDismissRequest = {
                showTips = false
                onDismissRequest()
            },
            title = stringResource(id = R.string.download_model)
        )

    AppDialog(onDismissRequest = onDismissRequest,
        title = { Text(stringResource(id = R.string.download_model)) }, content = {
            if (vm.error.isNotEmpty())
                Text(
                    text = vm.error,
                    modifier = Modifier.align(Alignment.Center),
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error
                )

            val isLoading = vm.modelList.isEmpty() && vm.error.isEmpty()
            LoadingContent(
                modifier = Modifier.fillMaxWidth(),
                isLoading = isLoading
            ) {
                Column(Modifier.fillMaxWidth()) {
                    val state = rememberLazyListState()
                    var search by rememberSaveable { mutableStateOf("") }
                    if (!isLoading && vm.error.isEmpty())
                        SearchTextFieldInList(
                            Modifier.align(Alignment.CenterHorizontally),
                            onSearch = { search = it }
                        )

                    LazyColumn(state = state) {
                        items(vm.modelList, key = { it.browserDownloadUrl }) { asset ->
                            if (!asset.name.contains(search, ignoreCase = true)) return@items

                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .clip(MaterialTheme.shapes.medium)
                                    .clickableRipple {
                                        context.startService(
                                            Intent(
                                                context,
                                                ModelManagerService::class.java
                                            ).apply {
                                                data = asset.browserDownloadUrl.toUri()
                                                putExtra(EXTRA_FILE_NAME, asset.name)
                                            }
                                        )
                                        showTips = true
                                    }
                                    .minimumInteractiveComponentSize()
                                    .padding(horizontal = 4.dp, vertical = 2.dp),
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column {
                                    Text(asset.name, style = MaterialTheme.typography.titleSmall)
                                    val size =
                                        remember(asset.size) { asset.size.formatFileSize(context) }
                                    Text(size, style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }
                }
            }
        }, buttons = {
            Row {
//                TextButton(onClick = { /*TODO*/ }) {
//                    Text(stringResource(id = android.R.string.ok))
//                }

                TextButton(onClick = onDismissRequest) {
                    Text(stringResource(id = android.R.string.cancel))
                }
            }
        })
}

