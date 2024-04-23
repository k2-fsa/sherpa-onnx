package com.k2fsa.sherpa.onnx.tts.engine

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech

fun getSampleText(lang: String): String {
    var text = ""
    when (lang) {
        "ara" -> {
            text = "هذا هو محرك تحويل النص إلى كلام باستخدام الجيل القادم من كالدي"
        }

        "ben" -> {
            text = "এটি একটি টেক্সট-টু-স্পীচ ইঞ্জিন যা পরবর্তী প্রজন্মের কালডি ব্যবহার করে"
        }

        "bul" -> {
            text =
                "Това е машина за преобразуване на текст в реч, използваща Kaldi от следващо поколение"
        }

        "cat" -> {
            text = "Aquest és un motor de text a veu que utilitza Kaldi de nova generació"
        }

        "ces" -> {
            text = "Toto je převodník textu na řeč využívající novou generaci kaldi"
        }

        "dan" -> {
            text = "Dette er en tekst til tale-motor, der bruger næste generation af kaldi"
        }

        "deu" -> {
            text =
                "Dies ist eine Text-to-Speech-Engine, die Kaldi der nächsten Generation verwendet"
        }

        "ell" -> {
            text = "Αυτή είναι μια μηχανή κειμένου σε ομιλία που χρησιμοποιεί kaldi επόμενης γενιάς"
        }

        "eng" -> {
            text = "This is a text-to-speech engine using next generation Kaldi"
        }

        "est" -> {
            text = "See on teksti kõneks muutmise mootor, mis kasutab järgmise põlvkonna Kaldi"
        }

        "fin" -> {
            text = "Tämä on tekstistä puheeksi -moottori, joka käyttää seuraavan sukupolven kaldia"
        }

        "fra" -> {
            text = "Il s'agit d'un moteur de synthèse vocale utilisant Kaldi de nouvelle génération"
        }

        "gle" -> {
            text = "Is inneall téacs-go-hurlabhra é seo a úsáideann Kaldi den chéad ghlúin eile"
        }

        "hrv" -> {
            text =
                "Ovo je mehanizam za pretvaranje teksta u govor koji koristi Kaldi sljedeće generacije"
        }

        "hun" -> {
            text = "Ez egy szövegfelolvasó motor a következő generációs kaldi használatával"
        }

        "isl" -> {
            text = "Þetta er texta í tal vél sem notar næstu kynslóð kaldi"
        }

        "ita" -> {
            text = "Questo è un motore di sintesi vocale che utilizza kaldi di nuova generazione"
        }

        "kat" -> {
            text = "ეს არის ტექსტიდან მეტყველების ძრავა შემდეგი თაობის კალდის გამოყენებით"
        }

        "kaz" -> {
            text = "Бұл келесі буын kaldi көмегімен мәтіннен сөйлеуге арналған қозғалтқыш"
        }

        "mlt" -> {
            text = "Din hija magna text-to-speech li tuża Kaldi tal-ġenerazzjoni li jmiss"
        }

        "lav" -> {
            text = "Šis ir teksta pārvēršanas runā dzinējs, kas izmanto nākamās paaudzes Kaldi"
        }

        "lit" -> {
            text = "Tai teksto į kalbą variklis, kuriame naudojamas naujos kartos Kaldi"
        }

        "ltz" -> {
            text = "Dëst ass en Text-zu-Speech-Motor mat der nächster Generatioun Kaldi"
        }

        "nep" -> {
            text = "यो अर्को पुस्ता काल्डी प्रयोग गरेर स्पीच इन्जिनको पाठ हो"
        }

        "nld" -> {
            text =
                "Dit is een tekst-naar-spraak-engine die gebruik maakt van Kaldi van de volgende generatie"
        }

        "nor" -> {
            text = "Dette er en tekst til tale-motor som bruker neste generasjons kaldi"
        }

        "pol" -> {
            text = "Jest to silnik syntezatora mowy wykorzystujący Kaldi nowej generacji"
        }

        "por" -> {
            text =
                "Este é um mecanismo de conversão de texto em fala usando Kaldi de próxima geração"
        }

        "ron" -> {
            text = "Acesta este un motor text to speech care folosește generația următoare de kadi"
        }

        "rus" -> {
            text =
                "Это движок преобразования текста в речь, использующий Kaldi следующего поколения."
        }

        "slk" -> {
            text = "Toto je nástroj na prevod textu na reč využívajúci kaldi novej generácie"
        }

        "slv" -> {
            text =
                "To je mehanizem za pretvorbo besedila v govor, ki uporablja Kaldi naslednje generacije"
        }

        "spa" -> {
            text = "Este es un motor de texto a voz que utiliza kaldi de próxima generación."
        }

        "srp" -> {
            text =
                "Ово је механизам за претварање текста у говор који користи калди следеће генерације"
        }

        "swa" -> {
            text = "Haya ni maandishi kwa injini ya hotuba kwa kutumia kizazi kijacho kaldi"
        }

        "swe" -> {
            text = "Detta är en text till tal-motor som använder nästa generations kaldi"
        }

        "tur" -> {
            text = "Bu, yeni nesil kaldi'yi kullanan bir metinden konuşmaya motorudur"
        }

        "ukr" -> {
            text =
                "Це механізм перетворення тексту на мовлення, який використовує kaldi нового покоління"
        }

        "vie" -> {
            text = "Đây là công cụ chuyển văn bản thành giọng nói sử dụng kaldi thế hệ tiếp theo"
        }

        "zho", "cmn" -> {
            text = "使用新一代卡尔迪的语音合成引擎"
        }
    }
    return text
}

class GetSampleText : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        var result = TextToSpeech.LANG_AVAILABLE
        val text: String = getSampleText(TtsEngine.lang ?: "")
        if (text.isEmpty()) {
            result = TextToSpeech.LANG_NOT_SUPPORTED
        }

        val intent = Intent().apply {
            if (result == TextToSpeech.LANG_AVAILABLE) {
                putExtra(TextToSpeech.Engine.EXTRA_SAMPLE_TEXT, text)
            } else {
                putExtra("sampleText", text)
            }
        }

        setResult(result, intent)
        finish()
    }
}