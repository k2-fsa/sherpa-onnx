package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import android.annotation.SuppressLint
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DeleteForever
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import com.k2fsa.sherpa.onnx.tts.engine.R

@Composable
fun DeleteForeverIcon(
    contentDescription: String? = stringResource(id = R.string.delete),
    @SuppressLint("ModifierParameter") modifier: Modifier = Modifier,
) {
    Icon(
        modifier = modifier,
        imageVector = Icons.Default.DeleteForever,
        tint = MaterialTheme.colorScheme.error,
        contentDescription = contentDescription,
    )
}