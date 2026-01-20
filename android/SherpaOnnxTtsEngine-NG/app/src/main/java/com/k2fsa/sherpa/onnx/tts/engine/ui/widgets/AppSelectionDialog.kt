package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
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
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.utils.ClipboardUtils
import com.k2fsa.sherpa.onnx.tts.engine.utils.clickableRipple
import com.k2fsa.sherpa.onnx.tts.engine.utils.performLongPress
import com.k2fsa.sherpa.onnx.tts.engine.utils.toast
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

@Composable
fun AppSelectionDialog(
    onDismissRequest: () -> Unit, title: @Composable () -> Unit,
    value: Any,
    values: List<Any>,
    entries: List<String>,
    isLoading: Boolean = false,
    searchEnabled: Boolean = values.size > 5,

    itemContent: @Composable RowScope.(Boolean, String, Any) -> Unit = { isSelected, entry, _ ->
        Text(
            entry,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(8.dp),
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
        )
    },

    buttons: @Composable BoxScope.() -> Unit = {
        TextButton(onClick = onDismissRequest) { Text(stringResource(id = android.R.string.cancel)) }
    },

    onValueSame: (Any, Any) -> Boolean = { a, b -> a == b },
    onClick: (Any, String) -> Unit,
) {
    val context = LocalContext.current
    val view = LocalView.current
    AppDialog(
        title = title,
        content = {
            val state = rememberLazyListState()
            LaunchedEffect(values) {
                val index = values.indexOfFirst { onValueSame(it, value) }
                if (index >= 0 && index < entries.size)
                    state.scrollToItem(index)
            }
            Column(modifier = Modifier.fillMaxWidth()) {
                var searchText by rememberSaveable { mutableStateOf("") }

                if (searchEnabled) {
                    val keyboardController = LocalSoftwareKeyboardController.current

                    var text by rememberSaveable { mutableStateOf("") }
                    DenseOutlinedField(
                        modifier = Modifier.align(Alignment.CenterHorizontally),
                        value = text, onValueChange = { text = it },
                        label = { Text(stringResource(id = R.string.search)) },
                        maxLines = 1,
                        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                        keyboardActions = KeyboardActions(
                            onDone = { keyboardController?.hide() }
                        )
                    )

                    LaunchedEffect(Unit) {
                        while (coroutineContext.isActive) {
                            delay(500)
                            searchText = text
                        }
                    }
                }

                val isEmpty by remember {
                    derivedStateOf { state.layoutInfo.viewportSize == IntSize.Zero }
                }

                if (searchText.isNotBlank() && isEmpty)
                    Text(
                        modifier = Modifier
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                            .minimumInteractiveComponentSize()
                            .align(Alignment.CenterHorizontally),
                        text = stringResource(id = R.string.list_is_empty),
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.error,
                        fontWeight = FontWeight.Bold,
                        style = MaterialTheme.typography.titleMedium
                    )

                LoadingContent(
                    modifier = Modifier.padding(vertical = 16.dp),
                    isLoading = isLoading
                ) {
                    LazyColumn(state = state) {
                        itemsIndexed(entries) { i, entry ->
                            if (searchEnabled && searchText.isNotBlank() &&
                                !entry.contains(searchText, ignoreCase = true)) return@itemsIndexed

                            val current = values[i]
                            val isSelected = onValueSame(value, current)
                            Row(
                                Modifier
                                    .fillMaxWidth()
                                    .clip(MaterialTheme.shapes.medium)
                                    .background(if (isSelected) MaterialTheme.colorScheme.primaryContainer else Color.Unspecified)
                                    .clickableRipple(
                                        onClick = { onClick(current, entry) },
                                        onLongClick = {
                                            view.performLongPress()
                                            ClipboardUtils.copyText(entry)
                                            context.toast(R.string.copied)
                                        }
                                    )
                                    .minimumInteractiveComponentSize(),
                            ) {
                                itemContent(isSelected, entry, value)
                            }

                        }
                    }
                }
            }
        },
        buttons = buttons, onDismissRequest = onDismissRequest,
    )
}