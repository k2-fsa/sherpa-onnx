#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)


import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf


def compute_features(samples):
    stft_config = knf.StftConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        center=True,
        window_type="hann",
    )
    knf_stft = knf.Stft(stft_config)
    stft_result = knf_stft(samples.tolist())
    real = np.array(stft_result.real, dtype=np.float32).reshape(
        stft_result.num_frames, -1
    )
    imag = np.array(stft_result.imag, dtype=np.float32).reshape(
        stft_result.num_frames, -1
    )

    mag = np.sqrt(real * real + imag * imag).astype(np.float32)

    mel_opts = knf.MelBanksOptions()
    mel_opts.num_bins = 100
    mel_opts.low_freq = 0
    mel_opts.high_freq = 24000 // 2
    mel_opts.is_librosa = True
    mel_opts.norm = ""
    mel_opts.use_slaney_mel_scale = False

    frame_opts = knf.FrameExtractionOptions()
    frame_opts.samp_freq = 24000
    #  frame_opts.frame_length_ms = 1024 * 1000 / 24000
    #  frame_opts.frame_shift_ms = 256 * 1000 / 24000

    mel_filters = knf.MelBanks(mel_opts, frame_opts)
    mel_features = np.zeros((mag.shape[0], 100))
    for i in range(mag.shape[0]):
        mel_features[i] = mel_filters.compute(mag[i])
    print("sum", np.sum(mel_features), np.mean(mel_features))

    mel_features = np.log(mel_features + 1e-10)
    return mel_features


class OnnxModel:
    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str):
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str):
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: np.ndarray,
        prompt_tokens: np.ndarray,
        prompt_features_len: np.ndarray,
        speed: np.ndarray,
    ) -> np.ndarray:
        out = self.text_encoder.run(
            [
                self.text_encoder.get_outputs()[0].name,
            ],
            {
                self.text_encoder.get_inputs()[0].name: tokens,
                self.text_encoder.get_inputs()[1].name: prompt_tokens,
                self.text_encoder.get_inputs()[2].name: prompt_features_len,
                self.text_encoder.get_inputs()[3].name: speed,
            },
        )
        return out[0]

    def run_fm_decoder(
        self,
        t: np.ndarray,
        x: np.ndarray,
        text_condition: np.ndarray,
        speech_condition: np.ndarray,
        guidance_scale: np.ndarray,
    ) -> np.ndarray:
        out = self.fm_decoder.run(
            [
                self.fm_decoder.get_outputs()[0].name,
            ],
            {
                self.fm_decoder.get_inputs()[0].name: t,
                self.fm_decoder.get_inputs()[1].name: x,
                self.fm_decoder.get_inputs()[2].name: text_condition,
                self.fm_decoder.get_inputs()[3].name: speech_condition,
                self.fm_decoder.get_inputs()[4].name: guidance_scale,
            },
        )
        return out[0]


class OnnxVocosModel:
    def __init__(
        self,
        filename: str,
    ):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts
        self.model = ort.InferenceSession(
            filename,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"vocos {self.model.get_modelmeta().custom_metadata_map}")

        print("----------vocos----------")
        for i in self.model.get_inputs():
            print(i)

        print("-----")

        for i in self.model.get_outputs():
            print(i)
        print()

    def __call__(self, x: np.ndarray):
        """
        Args:
          x: (N, feat_dim, num_frames)
        Returns:
          mag: (N, n_fft/2+1, num_frames)
          x: (N, n_fft/2+1, num_frames)
          y: (N, n_fft/2+1, num_frames)

        The complex spectrum is mag * (x + j*y)
        """
        assert x.ndim == 3, x.shape
        assert x.shape[0] == 1, x.shape

        mag, x, y = self.model.run(
            [
                self.model.get_outputs()[0].name,
                self.model.get_outputs()[1].name,
                self.model.get_outputs()[2].name,
            ],
            {
                self.model.get_inputs()[0].name: x,
            },
        )

        return mag, x, y


def get_phones(text):
    if text[-1] != ".":
        text += "."

    word2tokens = dict()
    with open("./lexicon.txt", encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            word = fields[0]
            tokens = fields[1:]
            word2tokens[word] = tokens

    token2id = dict()
    with open("./tokens.txt", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 1:
                token2id[" "] = int(fields[0])
            else:
                token2id[fields[0]] = int(fields[1])

    tokens = []
    for w in text:
        if w in word2tokens:
            tokens += word2tokens[w]
        else:
            tokens.append(w)
    ids = []
    for t in tokens:
        if t in token2id:
            ids.append(token2id[t])
        else:
            print(f"skip {t}")

    return ids


def compute_rms(features):
    return np.sqrt(np.mean(np.square(features)))


def get_timestamps(num_steps, t_shift=1):
    steps = np.linspace(0, 1, num_steps + 1)
    if t_shift != 1:
        steps = t_shift * steps / (1 + (t_shift - 1) * steps)

    return steps.tolist()


def trim_leading_silence_energy(samples, frame_size=2048, hop=512, energy_thresh=0.5):
    energies = [
        np.sum(np.abs(samples[i : i + frame_size]) ** 2)
        for i in range(0, len(samples) - frame_size, hop)
    ]
    #  print(energies)
    # First frame whose energy exceeds threshold
    frame_index = next((i for i, e in enumerate(energies) if e > energy_thresh), 0)
    frame_index = max(frame_index - 3, 0)
    start_sample = frame_index * hop
    return samples[start_sample:]


def main():
    vocoder = OnnxVocosModel("./vocos_24khz.onnx")

    prompt_text = "各位村民, 大家新年好! 近期, 湖北省武汉市等多个地区"
    prompt_wav_filename = "news-female.wav"

    prompt_text = "本台消息, 中共中央国务院, 近日印发关于构建数据基础制度, 更好发挥数据要素作用的意见."
    prompt_wav_filename = "news-female-2.wav"

    prompt_text = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系."
    prompt_wav_filename = "leijun-1.wav"

    prompt_ids = get_phones(prompt_text)

    text = "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中."

    ids = get_phones(text)

    data, sample_rate = sf.read(
        prompt_wav_filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    if sample_rate != 24000:
        import librosa

        samples = librosa.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=24000,
        )
        sample_rate = 24000

    assert len(samples.shape) == 1, samples.shape

    rms = compute_rms(samples)
    print("rms", rms)

    target_rms = 0.1
    if rms < target_rms:
        samples = samples * target_rms / rms
    new_rms = compute_rms(samples)

    print("new_rms", new_rms)

    prompt_features = compute_features(samples)
    print("features.shape", prompt_features.shape)

    feat_scale = 0.1
    prompt_features = prompt_features * feat_scale

    model = OnnxModel(
        text_encoder_path="./text_encoder_int8.onnx",
        fm_decoder_path="./fm_decoder_int8.onnx",
    )

    tokens = np.array([ids], dtype=np.int64)
    assert len(tokens.shape) == 2, tokens.shape

    prompt_tokens = np.array([prompt_ids], dtype=np.int64)
    assert len(prompt_tokens.shape) == 2, prompt_tokens.shape
    prompt_features_len = np.array(prompt_features.shape[0], dtype=np.int64)
    speed = np.array(1.0, dtype=np.float32)

    print(tokens.shape, prompt_tokens.shape, prompt_features_len)

    text_condition = model.run_text_encoder(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        prompt_features_len=prompt_features_len,
        speed=speed,
    )

    x = np.random.randn(*text_condition.shape).astype(np.float32)

    speech_condition = np.pad(
        prompt_features,
        pad_width=((0, x.shape[1] - prompt_features.shape[0]), (0, 0)),
        mode="constant",
        constant_values=0,
    )[None].astype(np.float32)

    print(speech_condition.shape, prompt_features.shape)

    guidance_scale = np.array(1.0, dtype=np.float32)

    num_steps = 8
    steps = get_timestamps(num_steps=num_steps, t_shift=0.5)
    for i in range(num_steps):
        t = np.array(steps[i], dtype=np.float32)
        v = model.run_fm_decoder(
            t=t,
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale,
        )
        x = x + v * (steps[i + 1] - steps[i])
    print("prompt_features", prompt_features.shape)
    x = x[:, prompt_features.shape[0] :]
    print("x", x.shape)

    x = x / feat_scale
    mel = x.transpose(0, 2, 1)
    mag, x, y = vocoder(mel)
    print("mag", mag.shape, x.shape, y.shape)

    stft_result = knf.StftResult(
        real=(mag * x)[0].transpose().reshape(-1).tolist(),
        imag=(mag * y)[0].transpose().reshape(-1).tolist(),
        num_frames=mag.shape[2],
    )
    config = knf.StftConfig(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window_type="hann",
        center=True,
        pad_mode="reflect",
        normalized=False,
    )
    istft = knf.IStft(config)
    audio_vocos = istft(stft_result)

    audio_vocos = np.array(audio_vocos)
    audio_vocos = trim_leading_silence_energy(audio_vocos)

    #  if rms < target_rms:
    #      audio_vocos = audio_vocos / target_rms * rms

    sf.write("generated.wav", audio_vocos, sample_rate, "PCM_16")


if __name__ == "__main__":
    main()
