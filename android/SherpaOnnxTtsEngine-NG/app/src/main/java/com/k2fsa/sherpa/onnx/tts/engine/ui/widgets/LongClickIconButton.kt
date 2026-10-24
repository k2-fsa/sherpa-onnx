package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.ripple.rememberRipple
import androidx.compose.material3.IconButtonColors
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.State
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.utils.performLongPress

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun LongClickIconButton(
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
    onLongClickLabel: String? = null,
    enabled: Boolean = true,
    colors: IconButtonColors = IconButtonDefaults.iconButtonColors(),
    interactionSource: MutableInteractionSource = remember { MutableInteractionSource() },
    content: @Composable () -> Unit
) {
    val stateLayerSize = 40.0.dp
    val context = LocalContext.current
    val view = LocalView.current
    Box(
        modifier = modifier
            .minimumInteractiveComponentSize()
            .size(stateLayerSize)
            .clip(CircleShape)
            .background(color = colors.mContainerColor(enabled).value)
            .combinedClickable(
                onClick = onClick,
                onLongClick = {
                    view.performLongPress()
                    onLongClick()
                },
                onLongClickLabel = onLongClickLabel,
                enabled = enabled,
                role = Role.Button,
                interactionSource = interactionSource,
                indication = rememberRipple(
                    bounded = false,
                    radius = stateLayerSize / 2
                )
            ),
        contentAlignment = Alignment.Center
    ) {
        val contentColor = colors.mContentColor(enabled).value
        CompositionLocalProvider(LocalContentColor provides contentColor, content = content)
        content()
    }
}

@Composable
internal fun IconButtonColors.mContainerColor(enabled: Boolean): State<Color> {
    return rememberUpdatedState(if (enabled) containerColor else disabledContainerColor)
}

@Composable
internal fun IconButtonColors.mContentColor(enabled: Boolean): State<Color> {
    return rememberUpdatedState(if (enabled) contentColor else disabledContentColor)
}