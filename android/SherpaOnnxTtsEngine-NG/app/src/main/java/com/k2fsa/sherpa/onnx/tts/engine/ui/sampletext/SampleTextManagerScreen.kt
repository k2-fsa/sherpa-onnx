package com.k2fsa.sherpa.onnx.tts.engine.ui.sampletext

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.DeleteForever
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.ui.ConfirmDeleteDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.LanguageSelectionDialog
import com.k2fsa.sherpa.onnx.tts.engine.utils.newLocaleFromCode

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SampleTextManagerScreen() {
    val vm: SampleTextMangerViewModel = viewModel()

    var showAddLanguage by remember { mutableStateOf(false) }
    if (showAddLanguage) {
        LanguageSelectionDialog(
            onDismissRequest = { showAddLanguage = false },
            language = "",
            filter = vm.languages
        ) {
            vm.addLanguage(it)
            showAddLanguage = false
        }
    }

    Scaffold(topBar = {
        TopAppBar(title = { Text(stringResource(id = R.string.sample_text)) }, actions = {
            IconButton(onClick = { showAddLanguage = true }) {
                Icon(Icons.Default.Add, stringResource(id = R.string.add))
            }
        })
    }) {
        SampleTextManagerContent(Modifier.padding(it), vm = vm)
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun SampleTextManagerContent(modifier: Modifier, vm: SampleTextMangerViewModel = viewModel()) {

    var showEditDialog by remember { mutableStateOf<String?>(null) }
    if (showEditDialog != null) {
        val code = showEditDialog!!
        SampleTextEditDialog(
            onDismissRequest = { showEditDialog = null },
            code = showEditDialog!!,
            list = vm.getList(code) ?: emptyList(),
            onConfirm = {
                vm.updateList(code, it)
                showEditDialog = null
            }
        )
    }

    var showDeleteDialog by remember { mutableStateOf<String?>(null) }
    if (showDeleteDialog != null) {
        val code = showDeleteDialog!!
        ConfirmDeleteDialog(
            onDismissRequest = { showDeleteDialog = null },
            name = code
        ) {
            vm.removeLanguage(code)
            showDeleteDialog = null
        }
    }

    LazyColumn(modifier) {
        items(vm.languages) {
            val locale = remember(it) { newLocaleFromCode(it) }
            val displayName = remember(locale) { locale.getDisplayName(locale) }
            LanguageItem(
                Modifier
                    .animateItemPlacement()
                    .fillMaxWidth()
                    .padding(4.dp),
                name = displayName,
                code = it,
                onClick = { showEditDialog = it },
                onDelete = { showDeleteDialog = it }
            )
        }
    }
}


@Composable
fun LanguageItem(
    modifier: Modifier,
    name: String,
    code: String,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    ElevatedCard(modifier = modifier, onClick = onClick) {
        Row(Modifier.padding(4.dp)) {
            Column(Modifier.weight(1f)) {
                Text(text = name, style = MaterialTheme.typography.titleMedium)
                Text(text = code, style = MaterialTheme.typography.bodyMedium)
            }

            IconButton(onClick = onDelete) {
                Icon(Icons.Default.DeleteForever, stringResource(id = R.string.delete))
            }
        }
    }
}
