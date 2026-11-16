package com.k2fsa.sherpa.onnx.fa

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow

object NavBarItems {
    val BarItems = listOf(
        BarItem(
            title = "Align",
            image = Icons.Filled.PlayArrow,
            route = "align",
        ),
        BarItem(
            title = "Info",
            image = Icons.Filled.Info,
            route = "info",
        ),
    )
}
