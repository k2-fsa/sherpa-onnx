package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.activity.compose.BackHandler
import androidx.compose.animation.Crossfade
import androidx.compose.foundation.layout.RowScope
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Deselect
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.ui.res.stringResource
import com.k2fsa.sherpa.onnx.tts.engine.R

class SelectionToolBarState(
    val onSelectAll: () -> Unit, val onSelectInvert: () -> Unit,
    val onSelectClear: () -> Unit
) {
    internal val selectedCount: MutableState<Int> = mutableIntStateOf(0)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun InternalSelectionBar(
    state: SelectionToolBarState,
    actions: @Composable RowScope.() -> Unit,
) {
    BackHandler {
        state.onSelectClear()
    }

    TopAppBar(
        navigationIcon = {
            IconButton(onClick = state.onSelectClear) {
                Icon(Icons.Default.Close, contentDescription = stringResource(id = R.string.close))
            }
        },
        title = { Text(state.selectedCount.value.toString()) }, actions = {
            AppTooltip(tooltip = stringResource(android.R.string.selectAll)) {
                IconButton(onClick = state.onSelectAll) {
                    Icon(Icons.Default.SelectAll, it)
                }
            }

            AppTooltip(tooltip = stringResource(id = R.string.invert_select)) {
                IconButton(onClick = state.onSelectInvert) {
                    Icon(Icons.Default.Deselect, it)
                }
            }
            actions()
        }
    )
}

@Composable
fun AppSelectionToolBar(
    state: SelectionToolBarState,
    mainBar: @Composable () -> Unit,
    selectionActions: @Composable RowScope.() -> Unit
) {
    Crossfade(targetState = state.selectedCount.value > 0, label = "") { selectMode ->
        if (selectMode) {
            InternalSelectionBar(state, actions = selectionActions)
        } else {
            mainBar()
        }
    }
}