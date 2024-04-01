package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.selected
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.R

@Composable
fun VerticalBar(
    modifier: Modifier = Modifier,
    enabled: Boolean,
    color: Color = MaterialTheme.colorScheme.primary
) {
    val context = LocalContext.current
    Box(
        modifier = modifier
            .background(
                color = if (enabled) color else Color.Transparent,
                shape = MaterialTheme.shapes.small
            )
            .width(4.dp)
            .height(32.dp)
            .semantics {
                this.stateDescription = if (enabled) context.getString(R.string.enabled)
                else ""
            }
    )
}

@Composable
fun SelectableCard(
    name: String,
    selected: Boolean,
    modifier: Modifier,
    content: @Composable() (ColumnScope.() -> Unit)
) {
    val context = LocalContext.current
    val color =
        if (selected) MaterialTheme.colorScheme.primary else Color.Unspecified
    Card(
        modifier = modifier
            .semantics {
                this.stateDescription = "$name ${if (!selected) "not" else ""} selected"
                this.selected = selected
            },
        colors = if (selected) CardDefaults.elevatedCardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(
                alpha = 0.5f
            )
        )
        else CardDefaults.elevatedCardColors(),
        border = if (selected) BorderStroke(1.dp, MaterialTheme.colorScheme.primary) else null,
        content = content
    )
}