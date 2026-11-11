# Kokoro 7M distills (Arabic + English)

Conversion scripts for oddadmix's two 7M-parameter Kokoro distills:

| upstream | language | sherpa-onnx bundle |
|---|---|---|
| [oddadmix/Nabra-7M-Distill](https://huggingface.co/oddadmix/Nabra-7M-Distill) | Arabic (MSA) | [marwanelamami/nabra-7m-distill-sherpa-onnx](https://huggingface.co/marwanelamami/nabra-7m-distill-sherpa-onnx) |
| [oddadmix/Kokoro-7M-Distill](https://huggingface.co/oddadmix/Kokoro-7M-Distill) | English (US) | [marwanelamami/kokoro-7m-distill-sherpa-onnx](https://huggingface.co/marwanelamami/kokoro-7m-distill-sherpa-onnx) |

Both are ~30 MB in FP32 and ~8.7 MB in INT8, against 82M/325 MB for stock
Kokoro. They use the standard Kokoro I/O contract, so `OfflineTtsKokoroModelConfig`
loads them unchanged:

```
input_ids  INT64  [batch, seq]
ref_s      FLOAT  [batch, 256]
speed      FLOAT  [1]
-> audio   FLOAT  [seq * 300]
```

Credit: the models are oddadmix's work. These scripts only convert, filter and
package them.

## Usage

```bash
./run.sh
```

Or step by step:

```bash
python export_onnx.py --repo oddadmix/Nabra-7M-Distill --dir nabra_7m \
    --weights kokoro_arabic_7m.pth --out model.fp32.onnx
python add_meta_data.py --model model.fp32.onnx --lang ar \
    --language "Arabic (Modern Standard)" --comment "Nabra-7M-Distill Arabic FP32"
python dynamic_quantization.py --src model.fp32.onnx --dst model.int8.onnx
```

## Three things worth knowing

**1. The iSTFT leaves image tones.** The ISTFTNet decoder produces narrow tones
at 4.8 kHz and 9.6 kHz (sr/5, 2·sr/5 for its 20-point inverse STFT). A 65-tap
FIR notch is baked into the graph by `add_meta_data.py`, so every runtime gets
filtered audio with no post-processing. Measured band ratios drop from
1.46 / 4.71 to 0.005 / 0.032.

**2. INT8 is safe here, INT4 is not.** Log-mel distance from FP32 is 0.069
(Arabic) and 0.074 (English). For calibration, on the same metric the 82M INT8
build scores 0.124 and was judged clean by ear, while the 82M INT4 build scores
0.284 and was judged noisy. INT8 is published; INT4 was not attempted.

**3. The front-end must match training — and for English, sherpa cannot yet
supply it.** See below.

## Front-end

Both models were distilled against a specific G2P and, at 7M parameters, have
no capacity to absorb a mismatch: feed them a different phoneme stream and they
mispronounce confidently even though the graph and weights are correct.

**Arabic** works with the built-in espeak front-end. `arabic_g2p.py` upstream is
itself an espeak wrapper, so sherpa's path lands in the same place — duration
delta is 0.0–2.3% on the sample sentences. The two MSA pharyngeals ʕ (ع) and
ħ (ح) live on free Kokoro embedding slots 7 and 8 and are carried in
`tokens.txt`, so they survive.

**English does not.** misaki, Kokoro's own front-end, uses single-character
diphthongs — `W` = /aʊ/, `A` = /eɪ/, `O` = /oʊ/ — that espeak-ng never emits.
Sherpa's built-in path therefore produces a token stream the model was never
trained on, and durations come out up to 37% short.

The official `lexicon-us-en.txt` from `kokoro-multi-lang-v1_0` *does* use those
symbols and maps cleanly onto this model's `tokens.txt` (all 178,401 entries
resolve, zero unknown phonemes). But `KokoroMultiLangLexicon` only consults the
lexicon when the voice string is empty:

```c++
// kokoro-multi-lang-lexicon.cc
if (!voice.empty()) {
  return ConvertTextToTokenIDsWithEspeak(text, voice);
}
```

and `voice` is read with a default:

```c++
// offline-tts-kokoro-model.cc
SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.voice, "voice", "en-us");
```

An absent `voice` key becomes `"en-us"`, and a present-but-empty one trips the
macro's validity check and exits. So for a single-language Kokoro model there
is no metadata that reaches the lexicon branch — it is only reachable for
multi-lingual models that pass no voice.

Until that is addressed upstream, use `g2p_frontend.py` (included here and in
the HF repo) and feed the token ids to the model directly. Both repos ship
`lexicon-us-en.txt` so the fix is a metadata change away once the runtime
allows it.

## Voice pack

Both students — including the English one — were conditioned on `af_msa.pt`
during distillation. Kokoro's `af_heart.pt` was never seen in training;
upstream measures WER 0.0525 → 0.0701 with it and the delivery sounds clipped
and hurried. `voices.bin` is packed from `af_msa.pt` in both bundles.

Index the style pack at `pack[len(phonemes) - 1]`, matching Kokoro's pipeline.
