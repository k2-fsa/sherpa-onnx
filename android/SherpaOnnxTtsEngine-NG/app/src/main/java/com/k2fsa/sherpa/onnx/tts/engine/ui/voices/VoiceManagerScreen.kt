package com.k2fsa.sherpa.onnx.tts.engine.ui.voices

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.EditNote
import androidx.compose.material.icons.filled.Headset
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.SortByAlpha
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.conf.TtsConfig
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Voice
import com.k2fsa.sherpa.onnx.tts.engine.ui.AuditionDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.ConfirmDeleteDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.ErrorHandler
import com.k2fsa.sherpa.onnx.tts.engine.ui.ShadowReorderableItem
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppSelectionToolBar
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.DeleteMenuItem
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.SelectableCard
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.SelectionToolBarState
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.TextFieldDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.VerticalBar
import com.k2fsa.sherpa.onnx.tts.engine.utils.performLongPress
import com.k2fsa.sherpa.onnx.tts.engine.utils.toast
import org.burnoutcrew.reorderable.detectReorder
import org.burnoutcrew.reorderable.rememberReorderableLazyListState
import org.burnoutcrew.reorderable.reorderable

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun VoiceManagerScreen() {
    val vm: VoiceManagerViewModel = viewModel()
    val context = LocalContext.current

    var showAddVoiceDialog by remember { mutableStateOf<Voice?>(null) }
    if (showAddVoiceDialog != null) {
        AddVoiceDialog(
            onDismissRequest = { showAddVoiceDialog = null },
            initialVoice = showAddVoiceDialog!!,
            onConfirm = {
                showAddVoiceDialog = null
                vm.addVoice(it)
            }
        )
    }

    var showEditNameDialog by remember { mutableStateOf<Voice?>(null) }
    if (showEditNameDialog != null) {
        val voice = showEditNameDialog!!
        TextFieldDialog(
            title = stringResource(id = R.string.display_name),
            initialText = voice.name,
            onDismissRequest = { showEditNameDialog = null }) {
            showEditNameDialog = null
            vm.updateVoice(voice.copy(name = it))
        }
    }

    var showDeleteDialog by remember { mutableStateOf<List<Voice>?>(null) }
    if (showDeleteDialog != null) {
        val voices = showDeleteDialog!!
        ConfirmDeleteDialog(
            onDismissRequest = { showDeleteDialog = null },
            name = voices.joinToString { it.name },
        ) {
            vm.delete(voices)
            showDeleteDialog = null
        }
    }


    var showAudition by remember { mutableStateOf<Voice?>(null) }
    if (showAudition != null) {
        AuditionDialog(onDismissRequest = { showAudition = null }, voice = showAudition!!)
    }

    val selectionState = remember {
        SelectionToolBarState(
            onSelectAll = vm::selectAll,
            onSelectInvert = vm::selectInvert,
            onSelectClear = vm::selectClear
        )
    }

    Scaffold(topBar = {
        AppSelectionToolBar(state = selectionState, mainBar = {
            TopAppBar(title = { Text(stringResource(id = R.string.app_name)) }, actions = {
                IconButton(onClick = {
                    showAddVoiceDialog = Voice.EMPTY
                }) {
                    Icon(Icons.Default.Add, stringResource(id = R.string.add_voice))
                }

                var showSortOptions by rememberSaveable { mutableStateOf(false) }
                IconButton(onClick = { showSortOptions = true }) {
                    Icon(Icons.Default.SortByAlpha, stringResource(id = R.string.sort))

                    DropdownMenu(
                        expanded = showSortOptions,
                        onDismissRequest = { showSortOptions = false }) {
                        DropdownMenuItem(
                            text = { Text(stringResource(R.string.sort_by_name)) },
                            onClick = {
                                showSortOptions = false
                                vm.sortByName()
                            }
                        )

                        DropdownMenuItem(
                            text = { Text(stringResource(R.string.sort_by_model)) },
                            onClick = {
                                showSortOptions = false
                                vm.sortByModel()
                            }
                        )
                    }
                }
            })
        }) {
            var showOptions by rememberSaveable { mutableStateOf(false) }
            IconButton(onClick = { showOptions = true }) {
                Icon(Icons.Default.MoreVert, stringResource(id = R.string.more_options))

                DropdownMenu(expanded = showOptions, onDismissRequest = { showOptions = false }) {
                    DeleteMenuItem {
                        showOptions = false
                        showDeleteDialog = vm.selects.toList()
                    }
                }
            }
        }
    }) { paddingValues ->
        LaunchedEffect(key1 = vm) {
            vm.load()
        }
        ErrorHandler(vm = vm)

        LaunchedEffect(key1 = vm.selects.size) {
            selectionState.selectedCount.value = vm.selects.size
        }

        val reorderState =
            rememberReorderableLazyListState(listState = vm.listState, onMove = { from, to ->
                vm.move(from.index, to.index)
            })

        if (vm.voices.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    text = stringResource(id = R.string.no_voices_tips),
                    style = MaterialTheme.typography.titleMedium
                )
            }
        }

        val isSelectMode = vm.selects.isNotEmpty()
        LazyColumn(
            Modifier
                .padding(paddingValues)
                .fillMaxSize()
                .reorderable(reorderState),
            state = reorderState.listState
        ) {
            items(vm.voices, key = { it.toString() }) { voice ->
                ShadowReorderableItem(reorderableState = reorderState, key = voice.toString()) {
                    val enabled = voice.contains(TtsConfig.voice.value)
                    val selected = vm.isSelected(voice)
                    val available = remember(voice) { vm.isModelAvailable(voice) }
                    LaunchedEffect(key1 = available) {
                        if (!available && voice.contains(TtsConfig.voice.value))
                            TtsConfig.voice.value = Voice.EMPTY
                    }

                    fun select() {
                        if (selected) vm.unselect(voice) else vm.select(voice)
                    }

                    Item(
                        modifier = Modifier
                            .animateItemPlacement()
                            .padding(4.dp),
                        reorderModifier = Modifier.detectReorder(reorderState),
                        available = available,
                        enabled = enabled,
                        selected = vm.isSelected(voice),
                        name = voice.name,
                        model = voice.model,
                        id = voice.id.toString(),
                        onClick = {
                            if (isSelectMode) select()
                            else {
                                if (available)
                                    TtsConfig.voice.value = voice
                                else
                                    context.toast(
                                        context.getString(R.string.model_not_found, voice.model)
                                    )
                            }
                        },
                        onLongClick = {
                            select()
                        },

                        onCopy = { showAddVoiceDialog = voice },
                        onDelete = { showDeleteDialog = listOf(voice) },
                        onAudition = { showAudition = voice },
                        onEditName = { showEditNameDialog = voice }
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun Item(
    modifier: Modifier,
    reorderModifier: Modifier,

    available: Boolean,
    enabled: Boolean,
    selected: Boolean,

    id: String,
    name: String,
    model: String,

    onClick: () -> Unit,
    onLongClick: () -> Unit,

    onEditName: () -> Unit,
    onAudition: () -> Unit,
    onCopy: () -> Unit,
    onDelete: () -> Unit,
) {
    val context = LocalContext.current
    val view = LocalView.current
    val color =
        if (available)
            if (enabled) MaterialTheme.colorScheme.primary else Color.Unspecified
        else MaterialTheme.colorScheme.error

    val tint = if (enabled) MaterialTheme.colorScheme.primary else LocalContentColor.current
    val iconButtonColors = IconButtonDefaults.iconButtonColors(contentColor = tint)

    SelectableCard(
        name = name,
        selected = selected,
        modifier
            .clip(CardDefaults.shape)
            .combinedClickable(
                onClick = {
                    onClick()
                },
                onLongClick = {
                    view.performLongPress()
                    onLongClick()
                }
            ),
    ) {
        Row(Modifier.padding(4.dp), verticalAlignment = Alignment.CenterVertically) {
            VerticalBar(enabled = enabled)

            Column(
                Modifier
                    .padding(start = 4.dp)
                    .weight(1f)
            ) {
                Text(text = name, style = MaterialTheme.typography.titleMedium, color = color)
                Row {
                    if (id != "0")
                        Text(
                            modifier = Modifier.padding(end = 4.dp),
                            text = id,
                            style = MaterialTheme.typography.bodyMedium,
                            color = color
                        )

                    Text(
                        modifier = Modifier.semantics {
                            contentDescription = if (available) model
                            else context.getString(R.string.model_not_found, model)
                        },
                        text = model,
                        style = MaterialTheme.typography.bodyMedium,
                        color = color
                    )
                }
            }

            Row {
                IconButton(enabled = available, onClick = onAudition, colors = iconButtonColors) {
                    Icon(Icons.Default.Headset, stringResource(id = R.string.audition))
                }

                var showOptions by rememberSaveable { mutableStateOf(false) }
                IconButton(
                    modifier = reorderModifier,
                    colors = iconButtonColors,
                    onClick = { showOptions = true }) {
                    Icon(
                        Icons.Default.MoreVert,
                        stringResource(id = R.string.more_options),
                    )

                    DropdownMenu(
                        expanded = showOptions,
                        onDismissRequest = { showOptions = false }) {
                        DropdownMenuItem(
                            text = { Text(stringResource(id = R.string.edit_name)) },
                            onClick = {
                                showOptions = false
                                onEditName()
                            },
                            leadingIcon = {
                                Icon(Icons.Default.EditNote, null)
                            }
                        )

                        DropdownMenuItem(
                            text = { Text(stringResource(id = android.R.string.copy)) },
                            onClick = {
                                showOptions = false
                                onCopy()
                            },
                            leadingIcon = {
                                Icon(Icons.Default.ContentCopy, null)
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