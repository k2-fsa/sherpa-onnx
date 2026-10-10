package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.stateDescription
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.R
import kotlinx.coroutines.delay

@Composable
fun LoadingContent(
    modifier: Modifier = Modifier,
    isLoading: Boolean,
    content: @Composable BoxScope.() -> Unit
) {
    val context = LocalContext.current
    Box(modifier) {
        Box(
            Modifier
                .wrapContentSize()
                .alpha(if (isLoading) 0.2f else 1f)
        ) {
            content()
        }

        AnimatedVisibility(
            visible = isLoading, modifier = Modifier
                .size(64.dp)
                .align(Alignment.Center)
        ) {
            CircularProgressIndicator(modifier = Modifier.semantics {
                stateDescription = context.getString(R.string.loading)
            }, strokeWidth = 8.dp)
        }
    }
}

@Preview
@Composable
fun PreviewLoadingContent() {
    MaterialTheme {
        var loading by remember { mutableStateOf(true) }
        LaunchedEffect(Unit) {
            delay(3000)
            loading = false
        }

        LoadingContent(Modifier, loading) {
            OutlinedTextField(value = "hello", onValueChange = {}, label = { Text("Label") })
        }
    }
}