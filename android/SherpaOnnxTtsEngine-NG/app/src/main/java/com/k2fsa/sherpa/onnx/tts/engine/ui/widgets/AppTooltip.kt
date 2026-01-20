package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.PlainTooltip
import androidx.compose.material3.Text
import androidx.compose.material3.TooltipBox
import androidx.compose.material3.TooltipDefaults
import androidx.compose.material3.rememberTooltipState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalView
import com.k2fsa.sherpa.onnx.tts.engine.utils.performLongPress

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppTooltip(
    modifier: Modifier = Modifier,
    tooltip: String,
    content: @Composable (tooltip: String) -> Unit
) {
    val state = rememberTooltipState()
    TooltipBox(
        modifier = modifier,
        positionProvider = TooltipDefaults.rememberPlainTooltipPositionProvider(),
        tooltip = {
            PlainTooltip {
                val view = LocalView.current
                LaunchedEffect(key1 = Unit) {
                    view.performLongPress()
                }
                Text(tooltip)
            }
        },
        state = state,
        content = { content(tooltip) },
    )
}