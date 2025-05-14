package com.k2fsa.sherpa.onnx.simulate.streaming.asr

sealed class NavRoutes(val route: String) {
    object Home : NavRoutes("home")
    object Help : NavRoutes("help")
}