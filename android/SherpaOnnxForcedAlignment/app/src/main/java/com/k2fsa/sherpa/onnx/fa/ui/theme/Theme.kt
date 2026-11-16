package com.k2fsa.sherpa.onnx.fa.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

// Colors mirror tauri-examples/doubao-asr-file/src/styles.css
val Primary = Color(0xFF2563EB)
val Background = Color(0xFFF5F6F8)
val Card = Color(0xFFFFFFFF)
val TextDark = Color(0xFF1F2329)
val Muted = Color(0xFF6B7280)
val ErrorRed = Color(0xFFB91C1C)
val Border = Color(0xFFE5E7EB)

private val LightColors = lightColorScheme(
    primary = Primary,
    onPrimary = Color.White,
    background = Background,
    onBackground = TextDark,
    surface = Card,
    onSurface = TextDark,
    surfaceVariant = Card,
    onSurfaceVariant = TextDark,
    error = ErrorRed,
    outline = Border,
)

private val AppShapes = Shapes(
    extraSmall = RoundedCornerShape(6.dp),
    small = RoundedCornerShape(6.dp),
    medium = RoundedCornerShape(10.dp),
    large = RoundedCornerShape(10.dp),
    extraLarge = RoundedCornerShape(10.dp),
)

@Composable
fun SherpaOnnxForcedAlignmentTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColors,
        shapes = AppShapes,
        content = content,
    )
}
