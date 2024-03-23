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

                                var testTextContent = ""
                                
                                when(TtsEngine.lang) {
                                    "ara" -> {
                                        testTextContent = "هذا هو محرك تحويل النص إلى كلام باستخدام الجيل القادم من كالدي"
                                    }
                                    "cat" -> {
                                        testTextContent = "Aquest és un motor de testText a veu que utilitza Kaldi de nova generació"
                                    }
                                    "ces" -> {
                                        testTextContent = "Toto je převodník testTextu na řeč využívající novou generaci kaldi"
                                    }
                                    "dan" -> {
                                        testTextContent = "Dette er en tekst til tale-motor, der bruger næste generation af kaldi"
                                    }
                                    "deu" -> {
                                        testTextContent = "Dies ist eine testText-to-Speech-Engine, die Kaldi der nächsten Generation verwendet"
                                    }
                                    "ell" -> {
                                        testTextContent = "Αυτή είναι μια μηχανή κειμένου σε ομιλία που χρησιμοποιεί kaldi επόμενης γενιάς"
                                    }
                                    "eng" -> {
                                        testTextContent = "This is a testText-to-speech engine using next generation Kaldi"
                                    }
                                    "fas" -> {
                                        testTextContent = "این یک موتور تبدیل متن به گفتار است برپایه نسخه پیشگام کالدی"
                                    }
                                    "fin" -> {
                                        testTextContent = "Tämä on tekstistä puheeksi -moottori, joka käyttää seuraavan sukupolven kaldia"
                                    }
                                    "fra" -> {
                                        testTextContent = "Il s'agit d'un moteur de synthèse vocale utilisant Kaldi de nouvelle génération."
                                    }
                                    "hun" -> {
                                        testTextContent = "Ez egy szövegfelolvasó motor a következő generációs kaldi használatával"
                                    }
                                    "isl" -> {
                                        testTextContent = "Þetta er testTexta í tal vél sem notar næstu kynslóð kaldi"
                                    }
                                    "ita" -> {
                                        testTextContent = "Questo è un motore di sintesi vocale che utilizza kaldi di nuova generazione"
                                    }
                                    "kat" -> {
                                        testTextContent = "ეს არის ტექსტიდან მეტყველების ძრავა შემდეგი თაობის კალდის გამოყენებით"
                                    }
                                    "kaz" -> {
                                        testTextContent = "Бұл келесі буын kaldi көмегімен мәтіннен сөйлеуге арналған қозғалтқыш"
                                    }
                                    "ltz" -> {
                                        testTextContent = "Dëst ass en testText-zu-Speech-Motor mat der nächster Generatioun Kaldi"
                                    }
                                    "nep" -> {
                                        testTextContent = "यो अर्को पुस्ता काल्डी प्रयोग गरेर स्पीच इन्जिनको पाठ हो"
                                    }
                                    "nld" -> {
                                        testTextContent = "Dit is een tekst-naar-spraak-engine die gebruik maakt van Kaldi van de volgende generatie"
                                    }
                                    "nor" -> {
                                        testTextContent = "Dette er en tekst til tale-motor som bruker neste generasjons kaldi"
                                    }
                                    "pol" -> {
                                        testTextContent = "Jest to silnik syntezatora mowy wykorzystujący Kaldi nowej generacji"
                                    }
                                    "por" -> {
                                        testTextContent = "Este é um mecanismo de conversão de testTexto em fala usando Kaldi de próxima geração"
                                    }
                                    "ron" -> {
                                        testTextContent = "Acesta este un motor testText to speech care folosește generația următoare de kadi"
                                    }
                                    "rus" -> {
                                        testTextContent = "Это движок преобразования текста в речь, использующий Kaldi следующего поколения."
                                    }
                                    "slk" -> {
                                        testTextContent = "Toto je nástroj na prevod testTextu na reč využívajúci kaldi novej generácie"
                                    }
                                    "spa" -> {
                                        testTextContent = "Este es un motor de testTexto a voz que utiliza kaldi de próxima generación."
                                    }
                                    "srp" -> {
                                        testTextContent = "Ово је механизам за претварање текста у говор који користи калди следеће генерације"
                                    }
                                    "swa" -> {
                                        testTextContent = "Haya ni maandishi kwa injini ya hotuba kwa kutumia kizazi kijacho kaldi"
                                    }
                                    "swe" -> {
                                        testTextContent = "Detta är en testText till tal-motor som använder nästa generations kaldi"
                                    }
                                    "tur" -> {
                                        testTextContent = "Bu, yeni nesil kaldi'yi kullanan bir metinden konuşmaya motorudur"
                                    }
                                    "ukr" -> {
                                        testTextContent = "Це механізм перетворення тексту на мовлення, який використовує kaldi нового покоління"
                                    }
                                    "vie" -> {
                                        testTextContent = "Đây là công cụ chuyển văn bản thành giọng nói sử dụng kaldi thế hệ tiếp theo"
                                    }
                                    "zho", "cmn" -> {
                                        testTextContent = "使用新一代卡尔迪的语音合成引擎"
                                    }
                                    else -> {
                                        testTextContent = ""
                                    }
                                }
                                
                                var testText by remember { mutableStateOf(testTextContent) }
                                
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
