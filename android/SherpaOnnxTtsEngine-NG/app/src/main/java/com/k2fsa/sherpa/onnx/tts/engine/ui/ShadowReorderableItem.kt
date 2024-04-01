package com.k2fsa.sherpa.onnx.tts.engine.ui

import android.view.HapticFeedbackConstants
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.lazy.LazyItemScope
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.DefaultShadowColor
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import org.burnoutcrew.reorderable.ReorderableItem
import org.burnoutcrew.reorderable.ReorderableState

@Composable
fun LazyItemScope.ShadowReorderableItem(
    reorderableState: ReorderableState<*>,
    key: Any,
    content: @Composable LazyItemScope.(isDragging: Boolean) -> Unit
) {
    val view = LocalView.current
    ReorderableItem(reorderableState, key) { isDragging ->
        if (isDragging) {
            view.isHapticFeedbackEnabled = true
            view.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
        }

        val elevation =
            animateDpAsState(if (isDragging) 24.dp else 0.dp, label = "")
        Box(
            modifier = Modifier
                .shadow(
                    elevation.value,
                    ambientColor = MaterialTheme.colorScheme.onBackground,
                    spotColor = MaterialTheme.colorScheme.onBackground
                )
        ) {
            content(isDragging)
        }
    }
}