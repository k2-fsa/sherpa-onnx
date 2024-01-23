package com.k2fsa.sherpa.onnx.speaker.identification

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountCircle
import androidx.compose.material.icons.filled.Add
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
            title = "Register",
            image = Icons.Filled.Add,
            route = "register",
        ),
        BarItem(
            title = "View",
            image = Icons.Filled.AccountCircle,
            route = "view",
        ),
        BarItem(
            title = "Help",
            image = Icons.Filled.Info,
            route = "help",
        ),
    )
}