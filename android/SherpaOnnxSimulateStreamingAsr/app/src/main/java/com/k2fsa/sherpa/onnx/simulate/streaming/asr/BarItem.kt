package com.k2fsa.sherpa.onnx.simulate.streaming.asr

import androidx.compose.ui.graphics.vector.ImageVector

data class BarItem(
    val title: String,

    // see https://www.composables.com/icons
    // and
    // https://developer.android.com/reference/kotlin/androidx/compose/material/icons/filled/package-summary
    val image: ImageVector,
    val route: String,
)
