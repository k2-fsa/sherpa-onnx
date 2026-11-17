package com.k2fsa.sherpa.onnx.fa

sealed class NavRoutes(val route: String) {
    object Align : NavRoutes("align")
    object Info : NavRoutes("info")
}
