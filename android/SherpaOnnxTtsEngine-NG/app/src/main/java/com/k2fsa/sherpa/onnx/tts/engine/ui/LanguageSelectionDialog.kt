package com.k2fsa.sherpa.onnx.tts.engine.ui

import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.res.stringResource
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppSelectionDialog
import com.k2fsa.sherpa.onnx.tts.engine.utils.newLocaleFromCode
import com.k2fsa.sherpa.onnx.tts.engine.utils.toCode
import java.util.Locale

@Composable
fun LanguageSelectionDialog(
    onDismissRequest: () -> Unit,
    language: String,
    filter: List<String> = emptyList(),
    onLanguageSelected: (String) -> Unit
) {
    val isoLangCodes =
        remember {
            Locale.getAvailableLocales().toList().map { it.toCode() }.distinct().filterNot {
                filter.contains(it)
            }
        }

    AppSelectionDialog(
        onDismissRequest = onDismissRequest,
        title = { Text(text = stringResource(id = R.string.language)) },
        value = language,
        values = isoLangCodes,
        entries = isoLangCodes.map {
            val displayName = newLocaleFromCode(it).displayName

            displayName + "\n" + it
        },
        onClick = { value, _ ->
            onLanguageSelected((value as String))
        },
    )
}