#!/usr/bin/env python3
# Copyright      2026  Xiaomi Corp.        (authors: Fangjun Kuang)


import librosa
import numpy as np
import onnxruntime as ort


class BinTokenizer:
    def __init__(self, path):
        self.tokens = self._load(path)

    def _load(self, path):
        tokens = []
        with open(path, "rb") as f:
            data = f.read()

        i = 0
        while i < len(data):
            first = data[i]
            i += 1

            if first == 0:
                tokens.append(b"")  # store as bytes
                continue

            if first < 128:
                length = first
            else:
                second = data[i]
                i += 1
                length = (second * 128) + (first - 128)

            token_bytes = data[i : i + length]
            i += length
            tokens.append(token_bytes)  # store as bytes, do NOT decode here

        return tokens

    def decode(self, ids):
        # join bytes first, then decode as UTF-8
        byte_stream = b"".join(self.tokens[i] for i in ids if i < len(self.tokens))
        text = byte_stream.decode("utf-8", errors="replace")
        return text.replace("▁", " ").strip()


class OnnxModel:
    def __init__(self, encoder, decoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.session_opts = session_opts

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

        print(f"----{encoder} input----")
        for i in self.encoder.get_inputs():
            print(i)

        print(f"----{encoder} output----")

        for i in self.encoder.get_outputs():
            print(i)

        print(f"----{decoder} input----")
        for i in self.decoder.get_inputs():
            print(i)

        print(f"----{decoder} output----")

        for i in self.decoder.get_outputs():
            print(i)

        self.need_decoder_attention_mask = False

        for n in self.decoder.get_inputs():
            if "key_values" in n.name and not hasattr(self, "num_head"):
                self.num_head = n.shape[1]
                self.head_dim = n.shape[3]

            if "encoder_attention_mask" in n.name:
                self.need_decoder_attention_mask = True
        if self.need_decoder_attention_mask:
            # [ mask, ids, encoder_out, states, use_cache_branch]
            self.num_layers = (len(self.decoder.get_inputs()) - 4) // 4
        else:
            # [ ids, encoder_out, states, use_cache_branch]
            self.num_layers = (len(self.decoder.get_inputs()) - 3) // 4

        self.bos = 1
        self.eos = 2

    def get_decoder_init_states(self):
        states = []
        shape = [1, self.num_head, 0, self.head_dim]
        for i in range(self.num_layers):
            decoder_key = np.zeros(shape, dtype=np.float32)
            decoder_value = np.zeros(shape, dtype=np.float32)
            encoder_key = np.zeros(shape, dtype=np.float32)
            encoder_value = np.zeros(shape, dtype=np.float32)

            states.append(decoder_key)
            states.append(decoder_value)
            states.append(encoder_key)
            states.append(encoder_value)

        return states

    def run_encoder(self, audio):
        audio = audio[None, :]  # batch=1

        if len(self.encoder.get_inputs()) > 1:
            mask = np.ones_like(audio, dtype=np.int64)

            outputs = self.encoder.run(
                [
                    self.encoder.get_outputs()[0].name,
                ],
                {
                    self.encoder.get_inputs()[0].name: audio,
                    self.encoder.get_inputs()[1].name: mask,
                },
            )
        else:
            outputs = self.encoder.run(
                [
                    self.encoder.get_outputs()[0].name,
                ],
                {
                    self.encoder.get_inputs()[0].name: audio,
                },
            )
        return outputs[0]  # last_hidden_state

    def run_decoder(self, token_id, encoder_out, states):
        inputs = dict()
        if self.need_decoder_attention_mask:
            mask = np.ones((1, encoder_out.shape[1]), dtype=np.int64)
            inputs[self.decoder.get_inputs()[0].name] = mask

            inputs[self.decoder.get_inputs()[1].name] = np.array(
                [[token_id]], dtype=np.int64
            )
            inputs[self.decoder.get_inputs()[2].name] = encoder_out

            for i in range(len(states)):
                inputs[self.decoder.get_inputs()[3 + i].name] = states[i]

            inputs[self.decoder.get_inputs()[-1].name] = np.array(
                [token_id != self.bos], dtype=bool
            )
        else:
            inputs[self.decoder.get_inputs()[0].name] = np.array(
                [[token_id]], dtype=np.int64
            )
            inputs[self.decoder.get_inputs()[1].name] = encoder_out

            for i in range(len(states)):
                inputs[self.decoder.get_inputs()[2 + i].name] = states[i]

            inputs[self.decoder.get_inputs()[-1].name] = np.array(
                [token_id != self.bos], dtype=bool
            )

        outputs = self.decoder.run(None, inputs)

        logits = outputs[0]
        if token_id == self.bos:
            states = outputs[1:]
        else:
            for i in range(self.num_layers):
                states[4 * i + 0] = outputs[1 + 4 * i + 0]
                states[4 * i + 1] = outputs[1 + 4 * i + 1]

        return logits, states


def load_audio(filename):
    audio, sample_rate = librosa.load(filename, sr=16000)
    assert sample_rate == 16000, sample_rate
    assert len(audio.shape) == 1, audio.shape

    return np.ascontiguousarray(audio[: 8 * 16000])


def main():
    model = OnnxModel(
        encoder="./tiny/encoder_model.ort",
        decoder="./tiny/decoder_model_merged.ort",
        #
        #  encoder="./tiny-zh/encoder_model.onnx",
        #  decoder="./tiny-zh/decoder_model_merged.onnx",
        #
        #  encoder="./base-zh/encoder_model.ort",
        #  decoder="./base-zh/decoder_model_merged.ort",
    )
    samples = load_audio("./two_cities.wav")
    print("samples.shape", samples.shape)
    encoder_out = model.run_encoder(samples)
    print("encoder_out.shape", encoder_out.shape)
    states = model.get_decoder_init_states()
    tokens = []

    max_len = int(len(samples) / 16000 * 15)

    token_id = model.bos

    for step in range(max_len):
        logits, states = model.run_decoder(token_id, encoder_out, states)
        token_id = int(np.argmax(logits[0, 0]))
        if token_id == model.eos:
            break
        tokens.append(token_id)
    print(tokens)

    tokenizer = BinTokenizer("./base-zh/tokenizer.bin")
    text = tokenizer.decode(tokens)
    print("text", text)


if __name__ == "__main__":
    main()
