package com.k2fsa.sherpa.onnx.speaker.diarization.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun HelpScreen() {
    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.padding(8.dp)
        ) {
            Text(
                "This app accepts only 16kHz 16-bit 1-channel *.wav files. " +
                        "It has two arguments: Number of speakers and clustering threshold. " +
                        "If you know the actual number of speakers in the file, please set it. " +
                        "Otherwise, please set it to 0. In that case, you have to set the threshold. " +
                        "A larger threshold leads to fewer segmented speakers."
            )
            Spacer(modifier = Modifier.height(5.dp))
            Text("The speaker segmentation model is from " +
                "pyannote-audio (https://huggingface.co/pyannote/segmentation-3.0), "+
                 "whereas the embedding extractor model is from 3D-Speaker (https://github.com/modelscope/3D-Speaker)")
            Spacer(modifier = Modifier.height(5.dp))
            Text("Please see http://github.com/k2-fsa/sherpa-onnx ")
            Spacer(modifier = Modifier.height(5.dp))
            Text("Everything is open-sourced!", fontSize = 20.sp)
        }
    }
}
