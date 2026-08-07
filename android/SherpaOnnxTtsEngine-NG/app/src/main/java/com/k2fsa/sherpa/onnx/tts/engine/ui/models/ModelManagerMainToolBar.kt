package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AddToPhotos
import androidx.compose.material.icons.filled.Archive
import androidx.compose.material.icons.filled.Download
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import com.k2fsa.sherpa.onnx.tts.engine.R

@OptIn(ExperimentalMaterial3Api::class)
@Composable
internal fun ModelManagerMainToolBar(
    modifier: Modifier,
    onAddModels: () -> Unit,
    onImportModels: () -> Unit,
    onDownloadModels: () -> Unit
) {
    val context = LocalContext.current
    TopAppBar(
        modifier = modifier,
        title = { Text(stringResource(id = R.string.app_name)) },
        actions = {
            var showOptions by remember { mutableStateOf(false) }
            IconButton(onClick = { showOptions = true }) {
                Icon(Icons.Default.Add, stringResource(id = R.string.add))
                DropdownMenu(expanded = showOptions, onDismissRequest = { showOptions = false }) {
                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.add_models)) },
                        onClick = onAddModels,
                        leadingIcon = {
                            Icon(Icons.Default.AddToPhotos, null)
                        }
                    )

                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.import_model_package)) },
                        onClick = onImportModels,
                        leadingIcon = {
                            Icon(Icons.Default.Archive, null)
                        }
                    )

                    DropdownMenuItem(
                        text = { Text(stringResource(R.string.download_model)) },
                        onClick = onDownloadModels,
                        leadingIcon = {
                            Icon(Icons.Default.Download, null)
                        }
                    )
                }
            }

        }
    )
}

