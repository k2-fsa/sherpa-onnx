#!/usr/bin/env python3
# Copyright (c)  2026 zengyw
# Generate calibration configs (voice/text/lang) with diverse voices and text.

import json
import random
from collections import Counter

SUPPORTED_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
    "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]

SENTENCES = {
    "en": [
        "Hello world.",
        "How are you today?",
        "The sky is blue.",
        "I love machine learning.",
        "Python is awesome.",
        "Good morning everyone.",
        "Artificial intelligence is growing.",
        "Speech synthesis is fascinating.",
        "Neural networks are powerful.",
        "Text to speech converts text to audio.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps machines understand text.",
        "Deep learning has revolutionized artificial intelligence.",
        "Speech synthesis technology has advanced significantly.",
        "Neural voice cloning can replicate speaking styles.",
        "Text normalization is important for proper pronunciation.",
        "Voice assistants help us interact with technology naturally.",
        "Modern TTS systems use deep learning for high-quality speech.",
        "Human computer interaction has become more intuitive.",
    ],
    "es": [
        "Hola mundo.",
        "¿Cómo estás hoy?",
        "El cielo es azul.",
        "Me encanta el aprendizaje automático.",
        "Python es increíble.",
        "Buenos días a todos.",
        "La inteligencia artificial está creciendo.",
        "La síntesis de voz es fascinante.",
        "Las redes neuronales son poderosas.",
        "El texto a voz convierte texto en audio.",
        "El veloz marrón salta sobre el perro perezoso.",
        "El aprendizaje automático permite a las computadoras aprender.",
        "El procesamiento del lenguaje natural ayuda a las máquinas.",
        "El aprendizaje profundo ha revolucionado la inteligencia artificial.",
        "La tecnología de síntesis de voz ha avanzado significativamente.",
        "La clonación de voz neuronal puede replicar estilos de habla.",
        "La normalización de texto es importante para la pronunciación.",
        "Los asistentes de voz nos ayudan a interactuar con la tecnología.",
        "Los sistemas TTS modernos utilizan aprendizaje profundo.",
        "La interacción humano computadora se ha vuelto más intuitiva.",
    ],
    "pt": [
        "Olá mundo.",
        "Como você está hoje?",
        "O céu é azul.",
        "Eu amo aprendizado de máquina.",
        "Python é incrível.",
        "Bom dia a todos.",
        "A inteligência artificial está crescendo.",
        "A síntese de voz é fascinante.",
        "As redes neurais são poderosas.",
        "Texto para voz converte texto em áudio.",
        "A rápida raposa marrom salta sobre o cachorro preguiçoso.",
        "O aprendizado de máquina permite que computadores aprendam.",
        "O processamento de linguagem natural ajuda máquinas a entender.",
        "O aprendizado profundo revolucionou a inteligência artificial.",
        "A tecnologia de síntese de voz avançou significativamente.",
        "A clonagem de voz neural pode replicar estilos de fala.",
        "A normalização de texto é importante para pronúncia.",
        "Assistentes de voz nos ajudam a interagir com tecnologia.",
        "Sistemas TTS modernos usam aprendizado profundo para áudio.",
        "A interação humano computador tornou-se mais intuitiva.",
    ],
    "fr": [
        "Bonjour le monde.",
        "Comment allez-vous aujourd'hui?",
        "Le ciel est bleu.",
        "J'aime l'apprentissage automatique.",
        "Python est incroyable.",
        "Bonjour à tous.",
        "L'intelligence artificielle grandit.",
        "La synthèse vocale est fascinante.",
        "Les réseaux neuronaux sont puissants.",
        "Le texte en voix convertit le texte en audio.",
        "Le rapide renard brun saute par-dessus le chien paresseux.",
        "L'apprentissage automatique permet aux ordinateurs d'apprendre.",
        "Le traitement du langage naturel aide les machines à comprendre.",
        "L'apprentissage profond a révolutionné l'intelligence artificielle.",
        "La technologie de synthèse vocale a considérablement progressé.",
        "Le clonage vocal neuronal peut reproduire les styles de parole.",
        "La normalisation du texte est importante pour la prononciation.",
        "Les assistants vocaux nous aident à interagir avec la technologie.",
        "Les systèmes TTS modernes utilisent l'apprentissage profond.",
        "L'interaction homme machine est devenue plus intuitive.",
    ],
    "ko": [
        "안녕하세요 세계.",
        "오늘 어떻게 지내세요?",
        "하늘이 푸릅니다.",
        "기계학습을 사랑합니다.",
        "파이썬은 놀라워요.",
        "모든 분께 좋은 아침입니다.",
        "인공지능이 성장하고 있습니다.",
        "음성 합성은 매력적입니다.",
        "신경망은 강력합니다.",
        "텍스트 음성 변환이 텍스트를 오디오로 변환합니다.",
        "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
        "기계학습이 컴퓨터가 데이터로 학습할 수 있게 합니다.",
        "자연어 처리가 기계를 이해하도록 돕습니다.",
        "딥러닝이 인공지능을 혁신했습니다.",
        "음성 합성 기술이 크게 발전했습니다.",
        "음성 클로닝이 음성 스타일을 복제할 수 있습니다.",
        "텍스트 정규화가 올바른 발음에 중요합니다.",
        "음성 비서가 기술과 상호작용하는 데 도움이 됩니다.",
        "최신 TTS 시스템이 고품질 음성을 생성합니다.",
        "인간 컴퓨터 상호작용이 더 직관적이 되었습니다.",
    ],
    "ja": [
        "こんにちは世界。",
        "今日はどのように過ごしていますか。",
        "空は青く、風は穏やかです。",
        "機械学習はデータから学ぶ技術です。",
        "音声合成は文章を自然な声に変換します。",
        "図書館では多くの人が静かに本を読んでいます。",
        "新しい列車の時刻表は来週から使われます。",
        "研究者たちは小さな端末で動くモデルを評価しました。",
        "音声アシスタントは毎日の作業を手伝います。",
        "天気予報によると午後から雨が降るそうです。",
    ],
    "ar": [
        "مرحبا بالعالم.",
        "كيف حالك اليوم؟",
        "السماء زرقاء والهواء لطيف.",
        "يساعد التعلم الآلي الحواسيب على فهم البيانات.",
        "تحول تقنية تحويل النص إلى كلام الجمل إلى صوت واضح.",
        "قرأ الطلاب قصة قصيرة في المكتبة صباحا.",
        "أعلن القطار عن تأخير بسيط بسبب أعمال الصيانة.",
        "تعمل النماذج الصغيرة بسرعة على الأجهزة المحلية.",
        "يساعد المساعد الصوتي المستخدمين في المهام اليومية.",
        "تحتاج الأنظمة الحديثة إلى قراءة مستقرة للنصوص الطويلة.",
    ],
    "bg": [
        "Здравей свят.",
        "Как си днес?",
        "Небето е синьо, а вятърът е тих.",
        "Машинното обучение помага на компютрите да учат от данни.",
        "Синтезът на реч превръща текст в ясен звук.",
        "Учениците прочетоха кратка история в библиотеката.",
        "Влакът закъсня заради поддръжка на релсите.",
        "Малките модели работят бързо на локални устройства.",
        "Гласовите асистенти улесняват ежедневните задачи.",
        "Стабилното четене е важно за дълги и кратки изречения.",
    ],
    "cs": [
        "Ahoj světe.",
        "Jak se dnes máš?",
        "Obloha je modrá a vítr je mírný.",
        "Strojové učení pomáhá počítačům učit se z dat.",
        "Syntéza řeči převádí text na srozumitelný zvuk.",
        "Studenti četli krátký příběh v knihovně.",
        "Vlak měl zpoždění kvůli údržbě trati.",
        "Malé modely běží rychle na místních zařízeních.",
        "Hlasový asistent pomáhá s každodenními úkoly.",
        "Stabilní čtení je důležité pro dlouhé i krátké věty.",
    ],
    "da": [
        "Hej verden.",
        "Hvordan har du det i dag?",
        "Himlen er blå, og vinden er mild.",
        "Maskinlæring hjælper computere med at lære af data.",
        "Talesyntese omdanner tekst til tydelig lyd.",
        "Eleverne læste en kort historie på biblioteket.",
        "Toget blev forsinket på grund af sporarbejde.",
        "Små modeller kører hurtigt på lokale enheder.",
        "En stemmeassistent hjælper med daglige opgaver.",
        "Stabil oplæsning er vigtig for både korte og lange sætninger.",
    ],
    "de": [
        "Hallo Welt.",
        "Wie geht es dir heute?",
        "Der Himmel ist blau und der Wind ist mild.",
        "Maschinelles Lernen hilft Computern, aus Daten zu lernen.",
        "Sprachsynthese wandelt Text in klare Sprache um.",
        "Die Schüler lasen am Morgen eine kurze Geschichte.",
        "Der Zug hatte wegen Wartungsarbeiten Verspätung.",
        "Kleine Modelle laufen schnell auf lokalen Geräten.",
        "Ein Sprachassistent hilft bei alltäglichen Aufgaben.",
        "Stabiles Vorlesen ist für kurze und lange Texte wichtig.",
    ],
    "el": [
        "Γεια σου κόσμε.",
        "Πώς είσαι σήμερα;",
        "Ο ουρανός είναι γαλάζιος και ο άνεμος ήρεμος.",
        "Η μηχανική μάθηση βοηθά τους υπολογιστές να μαθαίνουν από δεδομένα.",
        "Η σύνθεση ομιλίας μετατρέπει το κείμενο σε καθαρό ήχο.",
        "Οι μαθητές διάβασαν μια μικρή ιστορία στη βιβλιοθήκη.",
        "Το τρένο καθυστέρησε λόγω εργασιών συντήρησης.",
        "Τα μικρά μοντέλα λειτουργούν γρήγορα σε τοπικές συσκευές.",
        "Ο φωνητικός βοηθός διευκολύνει τις καθημερινές εργασίες.",
        "Η σταθερή ανάγνωση είναι σημαντική για σύντομα και μεγάλα κείμενα.",
    ],
    "et": [
        "Tere maailm.",
        "Kuidas sul täna läheb?",
        "Taevas on sinine ja tuul on vaikne.",
        "Masinõpe aitab arvutitel andmetest õppida.",
        "Kõnesüntees muudab teksti selgeks heliks.",
        "Õpilased lugesid raamatukogus lühikest lugu.",
        "Rong hilines rööbaste hoolduse tõttu.",
        "Väikesed mudelid töötavad kiiresti kohalikes seadmetes.",
        "Häälassistent aitab igapäevaste ülesannetega.",
        "Stabiilne lugemine on tähtis nii lühikeste kui pikkade lausete jaoks.",
    ],
    "fi": [
        "Hei maailma.",
        "Miten voit tänään?",
        "Taivas on sininen ja tuuli on lempeä.",
        "Koneoppiminen auttaa tietokoneita oppimaan datasta.",
        "Puhesynteesi muuttaa tekstin selkeäksi ääneksi.",
        "Oppilaat lukivat lyhyen tarinan kirjastossa.",
        "Juna myöhästyi raiteiden huollon vuoksi.",
        "Pienet mallit toimivat nopeasti paikallisilla laitteilla.",
        "Ääniavustaja auttaa päivittäisissä tehtävissä.",
        "Vakaa lukeminen on tärkeää sekä lyhyille että pitkille lauseille.",
    ],
    "hi": [
        "नमस्ते दुनिया.",
        "आज आप कैसे हैं?",
        "आसमान नीला है और हवा हल्की है.",
        "मशीन लर्निंग कंप्यूटरों को डेटा से सीखने में मदद करती है.",
        "वाक् संश्लेषण पाठ को स्पष्ट ध्वनि में बदलता है.",
        "छात्रों ने पुस्तकालय में एक छोटी कहानी पढ़ी.",
        "पटरियों की मरम्मत के कारण ट्रेन थोड़ी देर से आई.",
        "छोटे मॉडल स्थानीय उपकरणों पर तेज़ी से चलते हैं.",
        "वॉयस असिस्टेंट रोज़मर्रा के कामों में मदद करता है.",
        "लंबे और छोटे वाक्यों के लिए स्थिर पढ़ना महत्वपूर्ण है.",
    ],
    "hr": [
        "Pozdrav svijete.",
        "Kako si danas?",
        "Nebo je plavo, a vjetar je blag.",
        "Strojno učenje pomaže računalima učiti iz podataka.",
        "Sinteza govora pretvara tekst u jasan zvuk.",
        "Učenici su u knjižnici pročitali kratku priču.",
        "Vlak je kasnio zbog održavanja pruge.",
        "Mali modeli brzo rade na lokalnim uređajima.",
        "Glasovni asistent pomaže u svakodnevnim zadacima.",
        "Stabilno čitanje važno je za kratke i duge rečenice.",
    ],
    "hu": [
        "Helló világ.",
        "Hogy vagy ma?",
        "Az ég kék, a szél pedig enyhe.",
        "A gépi tanulás segít a számítógépeknek adatokból tanulni.",
        "A beszédszintézis a szöveget tiszta hanggá alakítja.",
        "A diákok rövid történetet olvastak a könyvtárban.",
        "A vonat a pálya karbantartása miatt késett.",
        "A kis modellek gyorsan futnak helyi eszközökön.",
        "A hangasszisztens segít a mindennapi feladatokban.",
        "A stabil felolvasás fontos rövid és hosszú mondatoknál is.",
    ],
    "id": [
        "Halo dunia.",
        "Apa kabar hari ini?",
        "Langit berwarna biru dan angin terasa lembut.",
        "Pembelajaran mesin membantu komputer belajar dari data.",
        "Sintesis ucapan mengubah teks menjadi suara yang jelas.",
        "Para siswa membaca cerita pendek di perpustakaan.",
        "Kereta terlambat karena perawatan rel.",
        "Model kecil berjalan cepat di perangkat lokal.",
        "Asisten suara membantu pekerjaan sehari-hari.",
        "Pembacaan yang stabil penting untuk kalimat pendek dan panjang.",
    ],
    "it": [
        "Ciao mondo.",
        "Come stai oggi?",
        "Il cielo è blu e il vento è leggero.",
        "L'apprendimento automatico aiuta i computer a imparare dai dati.",
        "La sintesi vocale trasforma il testo in audio chiaro.",
        "Gli studenti hanno letto una breve storia in biblioteca.",
        "Il treno ha subito un ritardo per lavori sui binari.",
        "I modelli piccoli funzionano rapidamente sui dispositivi locali.",
        "Un assistente vocale aiuta nelle attività quotidiane.",
        "Una lettura stabile è importante per frasi brevi e lunghe.",
    ],
    "lt": [
        "Labas pasauli.",
        "Kaip šiandien laikaisi?",
        "Dangus mėlynas, o vėjas švelnus.",
        "Mašininis mokymasis padeda kompiuteriams mokytis iš duomenų.",
        "Kalbos sintezė paverčia tekstą aiškiu garsu.",
        "Mokiniai bibliotekoje perskaitė trumpą istoriją.",
        "Traukinys vėlavo dėl bėgių priežiūros.",
        "Maži modeliai greitai veikia vietiniuose įrenginiuose.",
        "Balso asistentas padeda atlikti kasdienes užduotis.",
        "Stabilus skaitymas svarbus trumpiems ir ilgiems sakiniams.",
    ],
    "lv": [
        "Sveika pasaule.",
        "Kā tev šodien klājas?",
        "Debesis ir zilas, un vējš ir maigs.",
        "Mašīnmācīšanās palīdz datoriem mācīties no datiem.",
        "Runas sintēze pārvērš tekstu skaidrā skaņā.",
        "Skolēni bibliotēkā lasīja īsu stāstu.",
        "Vilciens kavējās sliežu remonta dēļ.",
        "Mazie modeļi ātri darbojas vietējās ierīcēs.",
        "Balss asistents palīdz ikdienas uzdevumos.",
        "Stabila lasīšana ir svarīga īsiem un gariem teikumiem.",
    ],
    "nl": [
        "Hallo wereld.",
        "Hoe gaat het vandaag?",
        "De lucht is blauw en de wind is zacht.",
        "Machine learning helpt computers om van gegevens te leren.",
        "Spraaksynthese zet tekst om in duidelijke audio.",
        "De leerlingen lazen een kort verhaal in de bibliotheek.",
        "De trein had vertraging door onderhoud aan het spoor.",
        "Kleine modellen draaien snel op lokale apparaten.",
        "Een stemassistent helpt bij dagelijkse taken.",
        "Stabiel voorlezen is belangrijk voor korte en lange zinnen.",
    ],
    "pl": [
        "Witaj świecie.",
        "Jak się dziś masz?",
        "Niebo jest niebieskie, a wiatr jest łagodny.",
        "Uczenie maszynowe pomaga komputerom uczyć się z danych.",
        "Synteza mowy zamienia tekst w wyraźny dźwięk.",
        "Uczniowie przeczytali krótką historię w bibliotece.",
        "Pociąg spóźnił się z powodu konserwacji torów.",
        "Małe modele działają szybko na lokalnych urządzeniach.",
        "Asystent głosowy pomaga w codziennych zadaniach.",
        "Stabilne czytanie jest ważne dla krótkich i długich zdań.",
    ],
    "ro": [
        "Salut lume.",
        "Cum te simți astăzi?",
        "Cerul este albastru, iar vântul este blând.",
        "Învățarea automată ajută computerele să învețe din date.",
        "Sinteza vocală transformă textul în sunet clar.",
        "Elevii au citit o poveste scurtă la bibliotecă.",
        "Trenul a întârziat din cauza lucrărilor la șine.",
        "Modelele mici rulează rapid pe dispozitive locale.",
        "Asistentul vocal ajută la sarcinile zilnice.",
        "Citirea stabilă este importantă pentru propoziții scurte și lungi.",
    ],
    "ru": [
        "Привет мир.",
        "Как у тебя дела сегодня?",
        "Небо голубое, а ветер мягкий.",
        "Машинное обучение помогает компьютерам учиться на данных.",
        "Синтез речи превращает текст в понятный звук.",
        "Ученики прочитали короткий рассказ в библиотеке.",
        "Поезд задержался из-за ремонта путей.",
        "Небольшие модели быстро работают на локальных устройствах.",
        "Голосовой помощник помогает в повседневных задачах.",
        "Стабильное чтение важно для коротких и длинных предложений.",
    ],
    "sk": [
        "Ahoj svet.",
        "Ako sa dnes máš?",
        "Obloha je modrá a vietor je mierny.",
        "Strojové učenie pomáha počítačom učiť sa z dát.",
        "Syntéza reči premieňa text na zrozumiteľný zvuk.",
        "Žiaci čítali krátky príbeh v knižnici.",
        "Vlak meškal pre údržbu trate.",
        "Malé modely bežia rýchlo na lokálnych zariadeniach.",
        "Hlasový asistent pomáha s každodennými úlohami.",
        "Stabilné čítanie je dôležité pre krátke aj dlhé vety.",
    ],
    "sl": [
        "Pozdravljen svet.",
        "Kako si danes?",
        "Nebo je modro in veter je nežen.",
        "Strojno učenje pomaga računalnikom učiti se iz podatkov.",
        "Sinteza govora pretvori besedilo v jasen zvok.",
        "Učenci so v knjižnici prebrali kratko zgodbo.",
        "Vlak je zamujal zaradi vzdrževanja tirov.",
        "Majhni modeli hitro delujejo na lokalnih napravah.",
        "Glasovni pomočnik pomaga pri vsakodnevnih opravilih.",
        "Stabilno branje je pomembno za kratke in dolge stavke.",
    ],
    "sv": [
        "Hej världen.",
        "Hur mår du idag?",
        "Himlen är blå och vinden är mild.",
        "Maskininlärning hjälper datorer att lära sig av data.",
        "Talsyntes omvandlar text till tydligt ljud.",
        "Eleverna läste en kort berättelse på biblioteket.",
        "Tåget blev försenat på grund av spårunderhåll.",
        "Små modeller kör snabbt på lokala enheter.",
        "En röstassistent hjälper till med vardagliga uppgifter.",
        "Stabil uppläsning är viktig för korta och långa meningar.",
    ],
    "tr": [
        "Merhaba dünya.",
        "Bugün nasılsın?",
        "Gökyüzü mavi ve rüzgar hafif.",
        "Makine öğrenimi bilgisayarların verilerden öğrenmesine yardımcı olur.",
        "Konuşma sentezi metni anlaşılır sese dönüştürür.",
        "Öğrenciler kütüphanede kısa bir hikaye okudu.",
        "Tren ray bakımı nedeniyle gecikti.",
        "Küçük modeller yerel cihazlarda hızlı çalışır.",
        "Sesli asistan günlük işlerde yardımcı olur.",
        "Kararlı okuma kısa ve uzun cümleler için önemlidir.",
    ],
    "uk": [
        "Привіт світе.",
        "Як ти сьогодні?",
        "Небо блакитне, а вітер лагідний.",
        "Машинне навчання допомагає комп'ютерам вчитися на даних.",
        "Синтез мовлення перетворює текст на зрозумілий звук.",
        "Учні прочитали коротку історію в бібліотеці.",
        "Потяг затримався через ремонт колії.",
        "Невеликі моделі швидко працюють на локальних пристроях.",
        "Голосовий помічник допомагає з щоденними завданнями.",
        "Стабільне читання важливе для коротких і довгих речень.",
    ],
    "vi": [
        "Xin chào thế giới.",
        "Hôm nay bạn thế nào?",
        "Bầu trời xanh và gió rất nhẹ.",
        "Học máy giúp máy tính học từ dữ liệu.",
        "Tổng hợp giọng nói chuyển văn bản thành âm thanh rõ ràng.",
        "Học sinh đọc một câu chuyện ngắn trong thư viện.",
        "Tàu bị trễ vì công việc bảo trì đường ray.",
        "Các mô hình nhỏ chạy nhanh trên thiết bị cục bộ.",
        "Trợ lý giọng nói hỗ trợ các công việc hằng ngày.",
        "Việc đọc ổn định rất quan trọng cho câu ngắn và câu dài.",
    ],
}

missing_langs = set(SUPPORTED_LANGS) - set(SENTENCES)
extra_langs = set(SENTENCES) - set(SUPPORTED_LANGS)
if missing_langs or extra_langs:
    raise RuntimeError(
        f"Calibration language mismatch. missing={sorted(missing_langs)}, "
        f"extra={sorted(extra_langs)}"
    )

VOICE_STYLES = {
    "M": [
        "assets/voice_styles/M1.json",
        "assets/voice_styles/M2.json",
        "assets/voice_styles/M3.json",
        "assets/voice_styles/M4.json",
        "assets/voice_styles/M5.json",
    ],
    "F": [
        "assets/voice_styles/F1.json",
        "assets/voice_styles/F2.json",
        "assets/voice_styles/F3.json",
        "assets/voice_styles/F4.json",
        "assets/voice_styles/F5.json",
    ],
}

SAMPLES_PER_LANG = 4


def generate_config():
    configs = []
    random.seed(42)
    voice_pool = VOICE_STYLES["M"] + VOICE_STYLES["F"]
    voice_offsets = {
        lang: random.randrange(len(voice_pool)) for lang in SUPPORTED_LANGS
    }

    for i in range(SAMPLES_PER_LANG):
        lang_order = SUPPORTED_LANGS[:]
        random.shuffle(lang_order)

        for lang in lang_order:
            sentences = SENTENCES[lang]
            voice = voice_pool[(voice_offsets[lang] + i) % len(voice_pool)]
            sentence_idx = i % len(sentences)
            sentence = sentences[sentence_idx]

            if i % 3 == 0:
                sentence2 = sentences[(sentence_idx + 1) % len(sentences)]
                sentence = sentence + " " + sentence2
            if i % 5 == 0:
                sentence3 = sentences[(sentence_idx + 2) % len(sentences)]
                sentence = sentence + " " + sentence3

            configs.append({
                "voice": voice,
                "text": sentence,
                "lang": lang,
            })

    return configs


def main():
    configs = generate_config()
    with open("calib_configs.json", "w", encoding="utf-8") as f:
        json.dump(configs, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(configs)} configurations saved to calib_configs.json")
    print("\nDistribution:")
    voices = [c["voice"].split("/")[-1] for c in configs]
    langs = [c["lang"] for c in configs]
    lens = [len(c["text"]) for c in configs]

    print("\nVoice distribution:")
    for v, c in Counter(voices).items():
        print(f"  {v}: {c}")

    print("\nLanguage distribution:")
    for lang, c in Counter(langs).items():
        print(f"  {lang}: {c}")

    print("\nText length stats:")
    print(f"  min: {min(lens)}, max: {max(lens)}, avg: {sum(lens)/len(lens):.1f}")

    print("\nSample configs:")
    for i in range(0, len(configs), len(SUPPORTED_LANGS)):
        c = configs[i]
        print(
            f"  [{i//len(SUPPORTED_LANGS) + 1}] lang={c['lang']}, "
            f"voice={c['voice'].split('/')[-1]}, text='{c['text'][:30]}...'"
        )


if __name__ == "__main__":
    main()
