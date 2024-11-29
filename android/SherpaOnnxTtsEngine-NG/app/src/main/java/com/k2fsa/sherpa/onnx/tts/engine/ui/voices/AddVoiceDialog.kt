package com.k2fsa.sherpa.onnx.tts.engine.ui.voices

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Headset
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.R
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager.toOfflineTtsConfig
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigVoiceManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.SynthesizerManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Voice
import com.k2fsa.sherpa.onnx.tts.engine.ui.AuditionDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppDialog
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.AppSpinner
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.CancelButton
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.DenseOutlinedField
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.IntSlider
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.LoadingContent
import com.k2fsa.sherpa.onnx.tts.engine.ui.widgets.OkButton
import com.k2fsa.sherpa.onnx.tts.engine.utils.performLongPress
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.internal.notify
import kotlin.math.max

@Composable
internal fun AddVoiceDialog(
    onDismissRequest: () -> Unit,
    initialVoice: Voice,
    onConfirm: (Voice) -> Unit
) {
    var voice by remember { mutableStateOf(initialVoice) }
    val isCopy = initialVoice != Voice.EMPTY

    val isDuplicate =
        remember(voice) {
            ConfigVoiceManager/*.filterNot { it == initialVoice }*/
                .find { it.toString() == voice.toString() } != null
        }
    var tips by remember { mutableStateOf("") }
    val context = LocalContext.current
    val view = LocalView.current
    LaunchedEffect(key1 = isDuplicate) {
        tips = if (isDuplicate) context.getString(R.string.duplicate_voices) else ""
        if (isDuplicate)
            (0..2).forEach { _ ->
                view.performLongPress()
            }
    }

    var showAudition by remember { mutableStateOf(false) }
    if (showAudition)
        AuditionDialog(onDismissRequest = { showAudition = false }, voice = voice)

    AppDialog(onDismissRequest = onDismissRequest, title = {
        Text(text = stringResource(if (isCopy) R.string.copy_voice else R.string.add_voice))
    }, content = {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            DenseOutlinedField(
                modifier = Modifier.fillMaxWidth(),
                value = voice.name,
                onValueChange = {
                    voice = voice.copy(name = it)
                },
                label = {
                    Text(text = stringResource(id = R.string.display_name))
                },
                isError = voice.name.isBlank()
            )

            val models = remember {
                ConfigModelManager.models().toMutableList().apply {
                    add(0, Model.EMPTY.copy(name = context.getString(R.string.please_select)))
                }
            }

            var loading by remember { mutableStateOf(true) }
            var speakerNum by remember { mutableIntStateOf(0) }
            AppSpinner(
                modifier = Modifier.padding(top = 4.dp),
                label = { Text(stringResource(id = R.string.model)) },
                value = voice.model,
                values = remember { models.map { it.id } },
                entries = remember { models.map { it.name } },
                onSelectedChange = { value, _ ->
                    val id = value as String
                    voice = voice.copy(model = id)
                }
            )

            LaunchedEffect(key1 = voice.model) {
                speakerNum = 0
                loading = true
                speakerNum = if (voice.model.isEmpty()) 1
                else withContext(Dispatchers.IO) {
                    models.find { it.id == voice.model }?.toOfflineTtsConfig()?.let {
                        val tts = SynthesizerManager.getTTS(it)
                        tts.numSpeakers()
                    } ?: 1
                }
                loading = false
            }

            val onlyDefault = speakerNum == 1
            LaunchedEffect(key1 = onlyDefault) {
                if (onlyDefault)
                    voice = voice.copy(id = 0)
            }

            LoadingContent(isLoading = loading) {
                val str = if (voice.id == 0) stringResource(id = R.string.default_speaker)
                else stringResource(R.string.speaker_id_desc, "${voice.id}")

                if (onlyDefault)
                    Text(
                        modifier = Modifier
                            .align(Alignment.Center)
                            .padding(vertical = 4.dp),
                        textAlign = TextAlign.Center,
                        text = stringResource(R.string.not_support_more_speaker),
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Bold
                    )
                else
                    IntSlider(
                        modifier = Modifier.padding(top = 4.dp),
                        label = str, value = voice.id.toFloat(), onValueChange = {
                            voice = voice.copy(id = it.toInt())
                        }, valueRange = 0f..max(speakerNum.toFloat(), 0f)
                    )
            }

            if (tips.isNotBlank())
                Text(
                    modifier = Modifier.padding(top = 4.dp),
                    text = tips,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                    fontWeight = FontWeight.Bold,
                )
        }
    }, buttons = {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Row(Modifier.weight(1f)) {
                IconButton(enabled = voice.model.isNotBlank(), onClick = { showAudition = true }) {
                    Icon(Icons.Default.Headset, stringResource(id = R.string.audition))
                }
            }
            CancelButton(onClick = onDismissRequest)
            OkButton(enabled = voice.name.isNotBlank() && voice.model.isNotBlank() && !isDuplicate) {
                onConfirm(voice)
                onDismissRequest()
            }
        }
    })
}