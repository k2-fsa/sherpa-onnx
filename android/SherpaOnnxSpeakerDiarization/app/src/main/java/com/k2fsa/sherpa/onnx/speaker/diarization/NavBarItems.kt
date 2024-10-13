package com.k2fsa.sherpa.onnx.speaker.diarization

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info

object NavBarItems {
    val BarItems = listOf(
        BarItem(
            title = "Home",
            image = Icons.Filled.Home,
            route = "home",
        ),
        BarItem(
            title = "Help",
            image = Icons.Filled.Info,
            route = "help",
        ),
    )
}