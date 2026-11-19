package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalTextInputService
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import kotlin.math.max

@Composable
@OptIn(ExperimentalMaterial3Api::class)
fun DropdownTextField(
    modifier: Modifier = Modifier,
    label: @Composable() (() -> Unit),
    value: Any,
    values: List<Any>,
    entries: List<String>,
    enabled: Boolean = true,
    leadingIcon: @Composable (() -> Unit)? = null,
    onValueSame: (current: Any, new: Any) -> Boolean = { current, new -> current == new },
    onSelectedChange: (value: Any, entry: String) -> Unit,
) {
    var selectedText = entries.getOrNull(max(0, values.indexOf(value))) ?: ""
    var expanded by remember { mutableStateOf(false) }

    LaunchedEffect(values, entries) {
        values.getOrNull(entries.indexOf(selectedText))?.let {
            onSelectedChange.invoke(it, selectedText)
        }
    }

    CompositionLocalProvider(
        LocalTextInputService provides null // Disable Keyboard
    ) {
        ExposedDropdownMenuBox(
            modifier = modifier,
            expanded = expanded,
            onExpandedChange = {
                if (enabled) expanded = !expanded
            },
        ) {
            OutlinedTextField(
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth(),
                leadingIcon = leadingIcon,
                readOnly = true,
                enabled = enabled,
                value = selectedText,
                onValueChange = { },
                label = label,
                trailingIcon = {
                    ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded)
                },
            )
            ExposedDropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                entries.forEachIndexed { index, text ->
                    val checked = onValueSame(value, values[index])
                    DropdownMenuItem(
                        text = {
                            Text(
                                text,
                                fontWeight = if (checked) FontWeight.Bold else FontWeight.Normal
                            )
                        },
                        onClick = {
                            expanded = false
                            selectedText = text
                            onSelectedChange.invoke(values[index], text)
                        }, modifier = Modifier.background(
                            if (checked) MaterialTheme.colorScheme.secondaryContainer
                            else Color.Unspecified
                        )
                    )
                }
            }
        }
    }
}


@Preview
@Composable
private fun PreviewDropdownTextField() {
    var key by remember { mutableIntStateOf(1) }
    AppSpinner(
        label = { Text("所属分组") },
        value = key,
        values = listOf(1, 2, 3),
        entries = listOf("1", "2", "3"),
    ) { k, _ ->
        key = k as Int
    }
}