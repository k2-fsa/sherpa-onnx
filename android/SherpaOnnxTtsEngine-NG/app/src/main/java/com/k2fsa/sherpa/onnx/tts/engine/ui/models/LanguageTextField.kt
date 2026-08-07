package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FilterList
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.ui.LanguageSelectionDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.DenseOutlinedField
import com.k2fsa.sherpa.onnx.tts.engine.utils.toLocale


@Composable
fun LanguageTextField(modifier: Modifier, language: String, onLanguageChange: (String) -> Unit) {
    var showLangSelectDialog by remember { mutableStateOf(false) }
    if (showLangSelectDialog)
        LanguageSelectionDialog(
            onDismissRequest = { showLangSelectDialog = false },
            language = language
        ) {
            onLanguageChange(it)
            showLangSelectDialog = false
        }

    Column(modifier) {
        DenseOutlinedField(
            modifier = modifier,
            value = language,
            onValueChange = onLanguageChange,
            label = { Text(text = stringResource(R.string.language)) },
            trailingIcon = {
                IconButton(onClick = { showLangSelectDialog = true }) {
                    Icon(
                        Icons.Default.FilterList,
                        contentDescription = stringResource(id = R.string.language)
                    )
                }
            }
        )

        val langName = remember(language) { language.toLocale().displayName }
        Text(
            modifier = Modifier.fillMaxWidth(),
            text = langName.ifBlank { language },
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center
        )
    }
}
