package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import com.k2fsa.sherpa.onnx.tts.engine.R

@Composable
fun DeleteMenuItem(modifier: Modifier = Modifier, onClick: () -> Unit) {
    DropdownMenuItem(
        modifier = modifier,
        text = {
            Text(
                stringResource(id = R.string.delete),
                color = MaterialTheme.colorScheme.error
            )
        },
        leadingIcon = { DeleteForeverIcon(null) },
        onClick = onClick
    )
}