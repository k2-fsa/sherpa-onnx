package com.k2fsa.sherpa.onnx.tts.engine.ui.widgets

import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import com.k2fsa.sherpa.onnx.tts.engine.R
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive

@Composable
fun SearchTextFieldInList(
    modifier: Modifier,
    onSearch: (String) -> Unit
) {
    val keyboardController = LocalSoftwareKeyboardController.current

    var search by rememberSaveable { mutableStateOf("") }

    var text by rememberSaveable { mutableStateOf("") }
    DenseOutlinedField(
        modifier = modifier,
        value = text, onValueChange = { text = it },
        label = { Text(stringResource(id = R.string.search)) },
        maxLines = 1,
        keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
        keyboardActions = KeyboardActions(
            onDone = { keyboardController?.hide() }
        )
    )

    LaunchedEffect(Unit) {
        while (coroutineContext.isActive) {
            delay(500)
            if (text != search) {
                search = text
                onSearch(search)
            }
        }
    }
}
