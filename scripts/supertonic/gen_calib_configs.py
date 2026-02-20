#!/usr/bin/env python3
# Copyright (c)  2026 zengyw

"""
Generate calibration configs (voice/text/lang) with diverse voices and text.
See also https://github.com/supertone-inc/supertonic
"""

import json
import random
from collections import Counter

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
}

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

LANGS = ["en", "es", "pt", "fr", "ko"]
SAMPLES_PER_LANG = 20


def generate_config():
    configs = []
    random.seed(42)

    for lang in LANGS:
        sentences = SENTENCES[lang]
        voice_pool = VOICE_STYLES["M"] + VOICE_STYLES["F"]
        random.shuffle(voice_pool)

        for i in range(SAMPLES_PER_LANG):
            voice = voice_pool[i % len(voice_pool)]
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

    random.shuffle(configs)
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
    for i in range(0, len(configs), 20):
        c = configs[i]
        print(f"  [{i//20 + 1}] lang={c['lang']}, voice={c['voice'].split('/')[-1]}, text='{c['text'][:30]}...'")


if __name__ == "__main__":
    main()
