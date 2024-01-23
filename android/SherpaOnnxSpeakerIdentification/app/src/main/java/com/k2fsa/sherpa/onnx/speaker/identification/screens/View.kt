package com.k2fsa.sherpa.onnx.speaker.identification.screens

import android.annotation.SuppressLint
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.toMutableStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.SpeakerRecognition

class SpeakerName(val name: String) {
    val nameState = mutableStateOf(name)
    val checked = mutableStateOf(false)

    fun onCheckedChange(newValue: Boolean) {
        checked.value = newValue
    }
}

@SuppressLint("UnrememberedMutableState")
@OptIn(ExperimentalFoundationApi::class)
@Composable
fun ViewScreen() {
    val allSpeakerNames = SpeakerRecognition.manager.allSpeakerNames()
    val allSpeakerNameList = remember {
        MutableList(
            allSpeakerNames.size
        ) {
            SpeakerName(allSpeakerNames[it])
        }.toMutableStateList()
    }

    var enabled by remember {
        mutableStateOf(SpeakerRecognition.manager.numSpeakers() > 0)
    }

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Button(
                enabled = enabled,
                onClick = {
                    val toRemove: MutableList<SpeakerName> = mutableListOf()
                    for (s in allSpeakerNameList) {
                        if (s.checked.value) {
                            SpeakerRecognition.manager.remove(s.name)
                            toRemove.add(s)
                        }
                    }
                    allSpeakerNameList.removeAll(toRemove)
                    enabled = SpeakerRecognition.manager.numSpeakers() > 0
                }) {
                Text("Delete selected")
            }
            LazyColumn(modifier = Modifier.fillMaxSize()) {
                items(allSpeakerNameList) { s: SpeakerName ->
                    ViewRow(speakerName = s)
                }
            }
        }
    }
}

@Composable
fun ViewRow(
    modifier: Modifier = Modifier,
    speakerName: SpeakerName
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(8.dp),
        color = MaterialTheme.colorScheme.inversePrimary,
    ) {
        Row(
            modifier = modifier,
            horizontalArrangement = Arrangement.Center,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = speakerName.name,
                modifier = modifier.weight(1.0F),
            )
            Checkbox(checked = speakerName.checked.value,
                onCheckedChange = { speakerName.onCheckedChange(it) }
            )
        }
    }
}