package com.k2fsa.sherpa.onnx.tts.engine.synthesizer

import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.SampleTextConfig
import kotlin.random.Random

object SampleTextManager {
    val defaultTextMap by lazy {
        hashMapOf(
            "zh-CN" to "使用新一代卡尔迪的语音合成引擎",
            "en-US" to "Use the new generation of Cardi's speech synthesis engine",
            "ja-JP" to "新しい世代のカーディの音声合成エンジンを使用する",
            "ko-KR" to "새로운 세대의 카디 음성 합성 엔진 사용",
            "fr-FR" to "Utilisez le nouveau moteur de synthèse vocale de Cardi",
            "de-DE" to "Verwenden Sie den neuen Cardi-Sprachsynthese-Engine",
            "es-ES" to "Utilice el nuevo motor de síntesis de voz de Cardi",
            "it-IT" to "Utilizza il nuovo motore di sintesi vocale di Cardi",
            "pt-BR" to "Use o novo motor de síntese de voz da Cardi",
            "ru-RU" to "Используйте новый двигатель синтеза речи Cardi",
            "ar-SA" to "استخدم محرك تركيب الصوت الجديد من Cardi",
            "tr-TR" to "Yeni nesil Cardi'nin konuşma sentezi motorunu kullanın",
            "vi-VN" to "Sử dụng động cơ tổng hợp giọng nói thế hệ mới của Cardi",
            "th-TH" to "ใช้เครื่องยนต์การสังเคราะห์เสียงรุ่นใหม่ของ Cardi",
            "id-ID" to "Gunakan mesin sintesis suara generasi baru Cardi",
            "ms-MY" to "Gunakan enjin sintesis suara generasi baru Cardi",
            "hi-IN" to "नए पीढ़ी के कार्डी की ध्वनि संश्लेषण इंजन का उपयोग करें",
            "bn-IN" to "নতুন প্রজন্মের কার্ডির বক্তব্য সিন্থেসিস ইঞ্জিন ব্যবহার করুন",
            "pa-IN" to "ਨਵੇਂ ਪੀੜੀ ਦੇ ਕਾਰਡੀ ਦੀ ਬੋਲੀ ਸੰਘਣਨ ਇੰਜਨ ਦੀ ਵਰਤੋਂ ਕਰੋ",
            "ta-IN" to "புதிய தலைப்பு கார்டியின் பேச்சு செயலி இயக்கத்தை பயன்படுத்தவும்",
            "te-IN" to "కార్డి యొక్క కూడా పీఠిక వాక్య సంయోజన ఇంజన్ ను ఉపయోగించండి",
            "ml-IN" to "കാർഡിയുടെ പുതിയ തലത്തിന്റെ സ്പീച് സിൻഥസിസ് ഇഞ്ചിൻ ഉപയോഗിക്കുക",
            "kn-IN" to "ಕಾರ್ಡಿಯ ಹೊಸ ತಲೆಯ ಸ್ಪೀಚ್ ಸಿಂಥಿಸೈಸ್ ಇಂಜಿನ್ ಅನ್ನು ಬಳಸಿ",
            "gu-IN" to "કાર્ડીની નવી પીઢીનો વાણી સંશ્લેષણ ઇંજન વાપરો",
            "mr-IN" to "कार्डीच्या नव्या पीढीचा ध्वनी संश्लेषण इंजिन वापरा",
            "or-IN" to "କାର୍ଡିର ନୂଆ ପୀଢ଼ୀର ବାକ୍ୟ ସଂଶ୍ଲେଷଣ ଇଞ୍ଜିନ ବ୍ୟବହାର କରନ୍ତୁ",
            "as-IN" to "কাৰ্ডীৰ নতুন পুংজীয়াৰ বাক্য সংশ্লেষণ ইঞ্জিন ব্যৱহাৰ কৰক",
            "ne-NP" to "कार्डीको नयाँ पीढीको वाणी संश्लेषण इन्जिन प्रयोग गर्नुहोस्",
            "si-LK" to "කාඩියේ නව පූර්ව වක්‍ර සංදර්ශක එන්ජින් භාවිතා කරන්න",
            "my-MM" to "ကာဒီ သည် အသစ်တစ်ခုအတွက် အသုံးပြုသည်",
            "km-KH" to "ប្រើប្រាស់ម៉ូតូសមុខរបស់កាតីក្នុងការបង្កើតសមុខរបស់ខ្លួន",
            "lo-LA" to "ໃຊ້ເຄື່ອງດຽວຂອງກາດອີໂອເຊຍໃໝ່",
        )

    }
    val defaultText: String
        get() = defaultTextMap["en-US"] ?: ""


    @Suppress("IfThenToElvis")
    fun getSampleText(code: String): String {
        val lang = code.lowercase()
        val texts = SampleTextConfig.config.toList()
        val list = texts.find { it.first.lowercase() == lang }?.second
            ?: texts.find { it.first.lowercase() == lang.substringBefore("-") }?.second
        return if (list == null) {
            defaultTextMap.toList().find { it.first.lowercase() == lang }?.second ?: defaultText
        } else {
            list.random(Random(System.currentTimeMillis()))
        }

    }
}