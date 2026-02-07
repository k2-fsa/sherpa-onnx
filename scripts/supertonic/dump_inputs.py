#!/usr/bin/env python3
# Copyright (c)  2026 zengyw

"""
Dump Supertonic TTS model inputs to npz for calibration.
See also https://github.com/supertone-inc/supertonic
"""

import argparse
import os

import numpy as np
import onnxruntime as ort

from helper import (
    UnicodeProcessor,
    Style,
    TextToSpeech,
    load_onnx_all,
    load_cfgs,
    load_text_processor,
    load_voice_style,
    chunk_text
)


class DumpTextToSpeech(TextToSpeech):
    """TTS with input dumping capability."""

    def __init__(
        self,
        cfgs: dict,
        text_processor: UnicodeProcessor,
        dp_ort: ort.InferenceSession,
        text_enc_ort: ort.InferenceSession,
        vector_est_ort: ort.InferenceSession,
        vocoder_ort: ort.InferenceSession,
        dump_dir: str = "calib",
    ):
        super().__init__(
            cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort
        )
        self.dump_dir = dump_dir

        self.dump_dirs = {
            "duration_predictor": os.path.join(dump_dir, "duration_predictor"),
            "text_encoder": os.path.join(dump_dir, "text_encoder"),
            "vector_estimator": os.path.join(dump_dir, "vector_estimator"),
            "vocoder": os.path.join(dump_dir, "vocoder"),
        }
        for d in self.dump_dirs.values():
            os.makedirs(d, exist_ok=True)
        self.counters = {k: 0 for k in self.dump_dirs}

    def _save_inputs(self, model_name: str, inputs: dict):
        """Save input tensors to npz file."""
        counter = self.counters[model_name]
        output_path = os.path.join(self.dump_dirs[model_name], f"{counter:03d}.npz")
        np.savez(output_path, **inputs)
        self.counters[model_name] += 1
        print(f"  Saved {model_name} inputs to {output_path}")

    def _infer(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference with input dumping."""
        assert (
            len(text_list) == style.ttl.shape[0]
        ), "Number of texts must match number of style vectors"
        bsz = len(text_list)

        text_ids, text_mask = self.text_processor(text_list, lang_list)
        dp_inputs = {
            "text_ids": text_ids,
            "style_dp": style.dp,
            "text_mask": text_mask,
        }
        self._save_inputs("duration_predictor", dp_inputs)
        dur_onnx, *_ = self.dp_ort.run(None, dp_inputs)
        dur_onnx = dur_onnx / speed
        text_emb_onnx, *_ = self.text_enc_ort.run(
            None,
            {
                "text_ids": text_ids,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
            },
        )
        self._save_inputs("text_encoder", {
            "text_ids": text_ids,
            "style_ttl": style.ttl,
            "text_mask": text_mask,
        })
        xt, latent_mask = self.sample_noisy_latent(dur_onnx)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)

        # dump vector_estimator inputs at last step (most informative)
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            ve_inputs = {
                "noisy_latent": xt,
                "text_emb": text_emb_onnx,
                "style_ttl": style.ttl,
                "text_mask": text_mask,
                "latent_mask": latent_mask,
                "current_step": current_step,
                "total_step": total_step_np,
            }
            if step == total_step - 1:
                self._save_inputs("vector_estimator", ve_inputs)
            xt, *_ = self.vector_est_ort.run(None, ve_inputs)

        # Vocoder inputs and run
        vocoder_inputs = {"latent": xt}
        self._save_inputs("vocoder", vocoder_inputs)
        wav, *_ = self.vocoder_ort.run(None, vocoder_inputs)

        return wav, dur_onnx

    def __call__(
        self,
        text: str,
        lang: str,
        style: Style,
        total_step: int,
        speed: float = 1.05,
        silence_duration: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single text to speech with input dumping."""
        assert (
            style.ttl.shape[0] == 1
        ), "Single speaker text to speech only supports single style"
        max_len = 120 if lang == "ko" else 300
        text_list = chunk_text(text, max_len=max_len)
        wav_cat = None
        dur_cat = None

        for i, text_chunk in enumerate(text_list):
            print(f"Processing chunk {i+1}/{len(text_list)}: '{text_chunk[:50]}...'")
            wav, dur_onnx = self._infer([text_chunk], [lang], style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur_onnx
            else:
                silence = np.zeros(
                    (1, int(silence_duration * self.sample_rate)), dtype=np.float32
                )
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat += dur_onnx + silence_duration
        return wav_cat, dur_cat

    def batch(
        self,
        text_list: list[str],
        lang_list: list[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batch inference with input dumping."""
        return self._infer(text_list, lang_list, style, total_step, speed)


def load_dump_text_to_speech(
    onnx_dir: str, dump_dir: str = "calib", use_gpu: bool = False
) -> DumpTextToSpeech:
    """Load TTS model for dumping inputs."""
    opts = ort.SessionOptions()
    if use_gpu:
        raise NotImplementedError("GPU mode is not fully tested")
    else:
        providers = ["CPUExecutionProvider"]
        print("Using CPU for inference")

    cfgs = load_cfgs(onnx_dir)
    dp_ort, text_enc_ort, vector_est_ort, vocoder_ort = load_onnx_all(
        onnx_dir, opts, providers
    )
    text_processor = load_text_processor(onnx_dir)
    return DumpTextToSpeech(
        cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort, dump_dir
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-dir", type=str, default="assets/onnx", help="onnx model dir"
    )
    parser.add_argument(
        "--dump-dir", type=str, default="calib", help="output npz dir"
    )
    parser.add_argument(
        "--total-step", type=int, default=5, help="denoising steps"
    )
    parser.add_argument(
        "--speed", type=float, default=1.05, help="speech speed"
    )
    parser.add_argument(
        "--n_test", type=int, default=1, help="num sentences"
    )
    parser.add_argument("--batch", action="store_true", help="batch mode")
    parser.add_argument(
        "--voice_style",
        type=str,
        nargs="+",
        default=["assets/voice_styles/M1.json"],
        help="voice style json path(s)",
    )
    parser.add_argument(
        "--text",
        type=str,
        nargs="+",
        default=[
            "This morning, I took a walk in the park, and the sound of the birds and the breeze was so pleasant."
        ],
        help="text(s) to synthesize",
    )
    parser.add_argument(
        "--lang", type=str, nargs="+", default=["en"], help="language(s)"
    )
    parser.add_argument("--clear", action="store_true", help="clear dump dir")
    parser.add_argument(
        "--config-file",
        type=str, default=None, dest="config_file", help="batch config json"
    )
    return parser.parse_args()


def main():
    args = get_args()

    if args.clear and os.path.exists(args.dump_dir):
        import shutil
        shutil.rmtree(args.dump_dir)
        print(f"Cleared existing directory: {args.dump_dir}")

    # Load TTS with dumping
    print(f"Loading models from {args.onnx_dir}...")

    if args.config_file:
        import json
        with open(args.config_file, "r") as f:
            configs = json.load(f)

        print(f"Loaded {len(configs)} configurations from {args.config_file}")

        # Process each configuration one by one
        tts = load_dump_text_to_speech(args.onnx_dir, args.dump_dir, use_gpu=False)

        print(f"\nProcessing {len(configs)} sentence(s)...")
        for i, cfg in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] voice={cfg['voice'].split('/')[-1]}, lang={cfg['lang']}")
            voice = load_voice_style([cfg["voice"]])
            _wav, _duration = tts(cfg["text"], cfg["lang"], voice, args.total_step, args.speed)
    else:
        # Validate inputs for non-batch mode
        if args.batch:
            assert len(args.voice_style) == len(args.text), (
                f"Number of voice styles ({len(args.voice_style)}) must match "
                f"number of texts ({len(args.text)})"
            )

        tts = load_dump_text_to_speech(args.onnx_dir, args.dump_dir, use_gpu=False)

        # Load voice style
        style = load_voice_style(args.voice_style, verbose=True)

        # Process sentences
        print(f"\nProcessing {args.n_test} sentence(s)...")
        for n in range(args.n_test):
            print(f"\n[{n+1}/{args.n_test}]")

            if args.batch:
                wav, duration = tts.batch(args.text, args.lang, style, args.total_step, args.speed)
            else:
                wav, duration = tts(args.text[0], args.lang[0], style, args.total_step, args.speed)

    # Print summary
    print("\n" + "=" * 50)
    print("Dumping completed!")
    print("=" * 50)
    print("\nGenerated files:")
    for model_name, counter in tts.counters.items():
        dump_dir = tts.dump_dirs[model_name]
        if os.path.exists(dump_dir):
            files = sorted(os.listdir(dump_dir))
            print(f"  {model_name}: {len(files)} files in {dump_dir}/")
            for f in files[:5]:
                print(f"    - {f}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")


if __name__ == "__main__":
    main()
