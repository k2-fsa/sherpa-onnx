@file:OptIn(ExperimentalMaterial3Api::class)

package com.k2fsa.sherpa.onnx.tts.engine

import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.ui.theme.SherpaOnnxTtsEngineTheme
import java.io.File
import java.lang.NumberFormatException

const val TAG = "sherpa-onnx-tts-engine"

class MainActivity : ComponentActivity() {
    // TODO(fangjun): Save settings in ttsViewModel
    private val ttsViewModel: TtsViewModel by viewModels()

    private var mediaPlayer: MediaPlayer? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        TtsEngine.createTts(this)
        setContent {
            SherpaOnnxTtsEngineTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Scaffold(topBar = {
                        TopAppBar(title = { Text("Next-gen Kaldi: TTS") })
                    }) {
                        Box(modifier = Modifier.padding(it)) {
                            Column(modifier = Modifier.padding(16.dp)) {
                                Column {
                                    Text("Speed " + String.format("%.1f", TtsEngine.speed))
                                    Slider(
                                        value = TtsEngine.speedState.value,
                                        onValueChange = { TtsEngine.speed = it },
                                        valueRange = 0.2F..3.0F,
                                        modifier = Modifier.fillMaxWidth()
                                    )
                                }
                                var testText by remember { mutableStateOf("") }

                                
                                when(TtsEngine.lang) {
                                    "ara" -> {
                                        testText = "هذا هو محرك تحويل النص إلى كلام باستخدام الجيل القادم من كالدي"
                                    }
                                    "cat" -> {
                                        testText = "Aquest és un motor de testText a veu que utilitza Kaldi de nova generació"
                                    }
                                    "ces" -> {
                                        testText = "Toto je převodník testTextu na řeč využívající novou generaci kaldi"
                                    }
                                    "dan" -> {
                                        testText = "Dette er en tekst til tale-motor, der bruger næste generation af kaldi"
                                    }
                                    "deu" -> {
                                        testText = "Dies ist eine testText-to-Speech-Engine, die Kaldi der nächsten Generation verwendet"
                                    }
                                    "ell" -> {
                                        testText = "Αυτή είναι μια μηχανή κειμένου σε ομιλία που χρησιμοποιεί kaldi επόμενης γενιάς"
                                    }
                                    "eng" -> {
                                        testText = "This is a testText-to-speech engine using next generation Kaldi"
                                    }
                                    "fas" -> {
                                        testText = "این یک موتور تبدیل متن به گفتار است برپایه نسخه پیشگام کالدی"
                                    }
                                    "fin" -> {
                                        testText = "Tämä on tekstistä puheeksi -moottori, joka käyttää seuraavan sukupolven kaldia"
                                    }
                                    "fra" -> {
                                        testText = "Il s'agit d'un moteur de synthèse vocale utilisant Kaldi de nouvelle génération."
                                    }
                                    "hun" -> {
                                        testText = "Ez egy szövegfelolvasó motor a következő generációs kaldi használatával"
                                    }
                                    "isl" -> {
                                        testText = "Þetta er testTexta í tal vél sem notar næstu kynslóð kaldi"
                                    }
                                    "ita" -> {
                                        testText = "Questo è un motore di sintesi vocale che utilizza kaldi di nuova generazione"
                                    }
                                    "kat" -> {
                                        testText = "ეს არის ტექსტიდან მეტყველების ძრავა შემდეგი თაობის კალდის გამოყენებით"
                                    }
                                    "kaz" -> {
                                        testText = "Бұл келесі буын kaldi көмегімен мәтіннен сөйлеуге арналған қозғалтқыш"
                                    }
                                    "ltz" -> {
                                        testText = "Dëst ass en testText-zu-Speech-Motor mat der nächster Generatioun Kaldi"
                                    }
                                    "nep" -> {
                                        testText = "यो अर्को पुस्ता काल्डी प्रयोग गरेर स्पीच इन्जिनको पाठ हो"
                                    }
                                    "nld" -> {
                                        testText = "Dit is een tekst-naar-spraak-engine die gebruik maakt van Kaldi van de volgende generatie"
                                    }
                                    "nor" -> {
                                        testText = "Dette er en tekst til tale-motor som bruker neste generasjons kaldi"
                                    }
                                    "pol" -> {
                                        testText = "Jest to silnik syntezatora mowy wykorzystujący Kaldi nowej generacji"
                                    }
                                    "por" -> {
                                        testText = "Este é um mecanismo de conversão de testTexto em fala usando Kaldi de próxima geração"
                                    }
                                    "ron" -> {
                                        testText = "Acesta este un motor testText to speech care folosește generația următoare de kadi"
                                    }
                                    "rus" -> {
                                        testText = "Это движок преобразования текста в речь, использующий Kaldi следующего поколения."
                                    }
                                    "slk" -> {
                                        testText = "Toto je nástroj na prevod testTextu na reč využívajúci kaldi novej generácie"
                                    }
                                    "spa" -> {
                                        testText = "Este es un motor de testTexto a voz que utiliza kaldi de próxima generación."
                                    }
                                    "srp" -> {
                                        testText = "Ово је механизам за претварање текста у говор који користи калди следеће генерације"
                                    }
                                    "swa" -> {
                                        testText = "Haya ni maandishi kwa injini ya hotuba kwa kutumia kizazi kijacho kaldi"
                                    }
                                    "swe" -> {
                                        testText = "Detta är en testText till tal-motor som använder nästa generations kaldi"
                                    }
                                    "tur" -> {
                                        testText = "Bu, yeni nesil kaldi'yi kullanan bir metinden konuşmaya motorudur"
                                    }
                                    "ukr" -> {
                                        testText = "Це механізм перетворення тексту на мовлення, який використовує kaldi нового покоління"
                                    }
                                    "vie" -> {
                                        testText = "Đây là công cụ chuyển văn bản thành giọng nói sử dụng kaldi thế hệ tiếp theo"
                                    }
                                    "zho", "cmn" -> {
                                        testText = "使用新一代卡尔迪的语音合成引擎"
                                    }
                                    else -> {
                                        testText = ""
                                    }
                                }

                                
                                val numSpeakers = TtsEngine.tts!!.numSpeakers()
                                if (numSpeakers > 1) {
                                    OutlinedTextField(
                                        value = TtsEngine.speakerIdState.value.toString(),
                                        onValueChange = {
                                            if (it.isEmpty() || it.isBlank()) {
                                                TtsEngine.speakerId = 0
                                            } else {
                                                try {
                                                    TtsEngine.speakerId = it.toString().toInt()
                                                } catch (ex: NumberFormatException) {
                                                    Log.i(TAG, "Invalid input: ${it}")
                                                    TtsEngine.speakerId = 0
                                                }
                                            }
                                        },
                                        label = {
                                            Text("Speaker ID: (0-${numSpeakers - 1})")
                                        },
                                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(bottom = 16.dp)
                                            .wrapContentHeight(),
                                    )
                                }

                                OutlinedTextField(
                                    value = testText,
                                    onValueChange = { testText = it },
                                    label = { Text("Please input your text here") },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(bottom = 16.dp)
                                        .wrapContentHeight(),
                                    singleLine = false,
                                )

                                Row {
                                    Button(
                                        modifier = Modifier.padding(20.dp),
                                        onClick = {
                                            Log.i(TAG, "Clicked, text: ${testText}")
                                            if (testText.isBlank() || testText.isEmpty()) {
                                                Toast.makeText(
                                                    applicationContext,
                                                    "Please input a test sentence",
                                                    Toast.LENGTH_SHORT
                                                ).show()
                                            } else {
                                                val audio = TtsEngine.tts!!.generate(
                                                    text = testText,
                                                    sid = TtsEngine.speakerId,
                                                    speed = TtsEngine.speed,
                                                )

                                                val filename =
                                                    application.filesDir.absolutePath + "/generated.wav"
                                                val ok =
                                                    audio.samples.size > 0 && audio.save(filename)

                                                if (ok) {
                                                    stopMediaPlayer()
                                                    mediaPlayer = MediaPlayer.create(
                                                        applicationContext,
                                                        Uri.fromFile(File(filename))
                                                    )
                                                    mediaPlayer?.start()
                                                } else {
                                                    Log.i(TAG, "Failed to generate or save audio")
                                                }
                                            }
                                        }) {
                                        Text("Test")
                                    }

                                    Button(
                                        modifier = Modifier.padding(20.dp),
                                        onClick = {
                                            TtsEngine.speakerId = 0
                                            TtsEngine.speed = 1.0f
                                            testText = ""
                                        }) {
                                        Text("Reset")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    override fun onDestroy() {
        stopMediaPlayer()
        super.onDestroy()
    }

    private fun stopMediaPlayer() {
        mediaPlayer?.stop()
        mediaPlayer?.release()
        mediaPlayer = null
    }
}
