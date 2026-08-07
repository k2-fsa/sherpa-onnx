package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Output
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.onLongClick
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import com.k2fsa.sherpa.onnx.tts.engine.ui.ErrorHandler
import com.k2fsa.sherpa.onnx.tts.engine.ui.ShadowReorderableItem
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppSelectionToolBar
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.DeleteMenuItem
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.SelectableCard
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.SelectionToolBarState
import com.k2fsa.sherpa.onnx.tts.engine.utils.performLongPress
import com.k2fsa.sherpa.onnx.tts.engine.utils.toLocale
import org.burnoutcrew.reorderable.detectReorder
import org.burnoutcrew.reorderable.rememberReorderableLazyListState
import org.burnoutcrew.reorderable.reorderable


@Composable
fun ModelManagerScreen() {
    val vm: ModelManagerViewModel = viewModel()
    val context = LocalContext.current
    var showImportDialog by remember { mutableStateOf(false) }
    if (showImportDialog)
        AddModelsDialog { showImportDialog = false }

    var showImportPackageDialog by remember { mutableStateOf(false) }
    if (showImportPackageDialog)
        ImportModelPackageDialog { showImportPackageDialog = false }

    var showDlModelDialog by remember { mutableStateOf(false) }
    if (showDlModelDialog)
        ModelDownloadInstallDialog { showDlModelDialog = false }

    var showLanguageDialog by remember { mutableStateOf(false) }
    if (showLanguageDialog) {
        var text by remember { mutableStateOf("") }
        AppDialog(
            onDismissRequest = { showLanguageDialog = false },
            title = { Text(stringResource(id = R.string.language)) },
            content = {
                LanguageTextField(modifier = Modifier.fillMaxWidth(), language = text) {
                    text = it
                }
            },
            buttons = {
                Row {
                    TextButton(onClick = { showLanguageDialog = false }) {
                        Text(stringResource(id = android.R.string.cancel))
                    }
                    TextButton(
                        enabled = text.isNotBlank(),
                        onClick = {
                            showLanguageDialog = false
                            vm.setLanguagesForSelectedModels(text)
                        }) {
                        Text(stringResource(id = android.R.string.ok))
                    }

                }
            }
        )
    }

    var showDeleteDialog by remember { mutableStateOf<List<Model>?>(null) }
    if (showDeleteDialog != null) {
        val models = showDeleteDialog!!
        ModelDeleteDialog(
            onDismissRequest = { showDeleteDialog = null },
            message = remember(models) { models.joinToString { it.name } },
            onConfirm = { deleteFile ->
                vm.deleteModels(models, deleteFile)
                showDeleteDialog = null
            }
        )
    }

    var showExportDialog by remember { mutableStateOf<List<Model>?>(null) }
    ModelExportDialog(
        show = showExportDialog != null,
        onDismissRequest = { showExportDialog = null },
        models = showExportDialog ?: emptyList(),
    )

    LaunchedEffect(key1 = vm) {
        vm.load()
    }
    ErrorHandler(vm = vm)

    val toolBarState = remember {
        SelectionToolBarState(
            onSelectAll = vm::selectAll,
            onSelectInvert = vm::selectInvert,
            onSelectClear = vm::clearSelect
        )
    }
    Scaffold(topBar = {
        AppSelectionToolBar(state = toolBarState, mainBar = {
            ModelManagerMainToolBar(
                modifier = Modifier,
                onAddModels = { showImportDialog = true },
                onImportModels = { showImportPackageDialog = true },
                onDownloadModels = { showDlModelDialog = true }
            )
        }) {
            var showOptions by remember { mutableStateOf(false) }
            IconButton(onClick = { showOptions = true }) {
                Icon(Icons.Default.MoreVert, stringResource(id = R.string.more_options))
                DropdownMenu(expanded = showOptions, onDismissRequest = { showOptions = false }) {
                    DropdownMenuItem(
                        leadingIcon = { Icon(Icons.Default.Output, null) },
                        text = { Text(stringResource(id = R.string.export)) },
                        onClick = {
                            showExportDialog = vm.selectedModels
                            showOptions = false
                        }
                    )

                    DropdownMenuItem(
                        leadingIcon = { Icon(Icons.Default.Language, null) },
                        text = { Text(stringResource(id = R.string.change_language)) },
                        onClick = {
                            showLanguageDialog = true
                            showOptions = false
                        }
                    )

                    DeleteMenuItem {
                        showOptions = false
                        showDeleteDialog = vm.selectedModels.toList()
                    }
                }
            }
        }
    }) {
        ModelManagerScreenContent(
            Modifier
                .padding(it)
                .fillMaxSize(),
            vm = vm,
            toolBarState = toolBarState,
            onDeleteModel = { showDeleteDialog = listOf(it) },
            onExportModel = { showExportDialog = listOf(it) }
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun ModelManagerScreenContent(
    modifier: Modifier = Modifier,
    toolBarState: SelectionToolBarState,
    onDeleteModel: (Model) -> Unit,
    onExportModel: (Model) -> Unit,

    vm: ModelManagerViewModel = viewModel()
) {
    val context = LocalContext.current

    var showModelEditDialog by remember { mutableStateOf<Model?>(null) }
    if (showModelEditDialog != null) {
        ModelEditDialog(
            onDismissRequest = { showModelEditDialog = null },
            model = showModelEditDialog!!,
            onSave = { ConfigModelManager.updateModels(it) }
        )
    }

    val selectMode = vm.selectedModels.isNotEmpty()
    LaunchedEffect(key1 = vm.selectedModels.size) {
        toolBarState.selectedCount.value = vm.selectedModels.size
    }

    val view = LocalView.current
    val reorderState =
        rememberReorderableLazyListState(listState = vm.listState, onMove = { from, to ->
            vm.moveModel(from.index, to.index)
            view.announceForAccessibility(
                context.getString(
                    R.string.list_moved_desc,
                    from.index.toString(),
                    to.index.toString()
                )
            )
        })

    if (vm.models.value.isEmpty()) {
        Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            Text(
                text = stringResource(id = R.string.no_models_tips),
                style = MaterialTheme.typography.titleMedium
            )
        }
    }

    LazyColumn(modifier = modifier.reorderable(reorderState), state = reorderState.listState) {
        items(vm.models.value, { it.id }) { model ->
            val lang = remember(model.lang) { model.lang.toLocale().displayName }
            val selected = vm.selectedModels.contains(model)
            ShadowReorderableItem(reorderableState = reorderState, key = model.id) {
                ModelItem(
                    modifier = Modifier
                        .animateItemPlacement()
                        .padding(4.dp),
                    reorderModifier = Modifier.detectReorder(reorderState),
                    name = model.name,
                    lang = lang,
                    selected = selected,
                    onEdit = { showModelEditDialog = model },
                    onClick = {
                        if (selectMode) {
                            if (selected)
                                vm.selectedModels.remove(model)
                            else
                                vm.selectedModels.add(model)
                        }
//                            TtsConfig.modelId.value = model.id
                    },

                    onLongClick = {
                        if (!selectMode)
                            vm.selectedModels.add(model)
                    },
                    onDelete = { onDeleteModel(model) },
                    onExport = { onExportModel(model) }
                )
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun ModelItem(
    modifier: Modifier,
    reorderModifier: Modifier = Modifier,
    name: String,
    lang: String,
    selected: Boolean,
    onExport: () -> Unit,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    onEdit: () -> Unit,
    onDelete: () -> Unit
) {
    val context = LocalContext.current
    val view = LocalView.current

    SelectableCard(
        name = name,
        selected = selected,
        modifier
            .clip(CardDefaults.shape)
            .combinedClickable(
                onClick = onClick,
                onLongClick = {
                    view.performLongPress()
                    onLongClick()
                },
            ),
    ) {
        Box(modifier = Modifier.padding(4.dp)) {
            Row {
                Row(
                    modifier = Modifier.weight(1f),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(
                        Modifier
                            .weight(1f)
                            .padding(start = 4.dp)
                    ) {
                        Text(
                            text = name,
                            style = MaterialTheme.typography.titleMedium,
                            maxLines = 1,

                            )
                        Row {
                            Text(
                                text = lang,
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                    Row {
                        IconButton(onClick = onEdit) {
                            Icon(
                                Icons.Default.Edit,
                                contentDescription = stringResource(id = R.string.edit),
                            )
                        }

                        var showOptions by remember { mutableStateOf(false) }
                        IconButton(modifier = reorderModifier.semantics {
                            onLongClick(context.getString(R.string.drag_sort_desc)) { true }
                        },
                            onClick = { showOptions = true }) {
                            Icon(
                                Icons.Default.MoreVert,
                                contentDescription = stringResource(id = R.string.more_options),
                            )

                            DropdownMenu(
                                expanded = showOptions,
                                onDismissRequest = { showOptions = false }) {

                                DropdownMenuItem(
                                    text = { Text(stringResource(id = R.string.export)) },
                                    leadingIcon = {
                                        Icon(Icons.Default.Output, null)
                                    },
                                    onClick = {
                                        showOptions = false
                                        onExport()
                                    }
                                )

                                DeleteMenuItem {
                                    showOptions = false
                                    onDelete()
                                }

                            }
                        }
                    }
                }
            }
        }
    }
}
