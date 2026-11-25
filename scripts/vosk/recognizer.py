#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/recognizer.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Python port of the sherpa-onnx recognizer design
# (sherpa-onnx/csrc/online-recognizer.h, offline-recognizer.h,
# online-stream.h, offline-stream.h): streams accept samples and compute
# features; recognizers test readiness and decode; the Decoder is a plain
# kaldi FST lattice decoder, fed one model-output row at a time.
#
#   OnlineStream       -- accepts samples chunk by chunk, computes MFCC +
#                         running i-vector (the feature pipeline)
#   OfflineStream      -- one utterance of samples -> features + i-vector
#   Decoder            -- kaldi LatticeFasterDecoder; fed one model-output
#                         row per decoder frame (a row every `factor`
#                         input feature frames)
#   OnlineRecognizer   -- create_stream / is_ready / decode_stream /
#                         get_result / reset
#   OfflineRecognizer  -- create_stream / decode_stream / get_result
#
# The recognizers take an AM backend (OnnxAm / TorchAm / FlatAm) -- the
# model is the only difference between the decode examples; everything
# else (streams, windowing, decoder) is shared.
#
# The streaming main loop mirrors the C++ microphone example:
#
#     rec = OnlineRecognizer(OnnxAm(model_dir), chunk_size=30)
#     stream = rec.create_stream()
#     for chunk in chunks:
#         stream.accept_waveform(sample_rate, chunk)
#         while rec.is_ready(stream):
#             rec.decode_stream(stream)
#         print(rec.get_result(stream))
#     stream.input_finished()
#     while rec.is_ready(stream):
#         rec.decode_stream(stream)
#     print(rec.get_result(stream))
#
# ("Is the stream ready?" -- the readiness rule and its rationale are
# documented on OnlineRecognizer.is_ready; the row bookkeeping it drives
# is documented on OnlineRecognizer.is_ready and decode_stream.)
#
# ALL bookkeeping counts INPUT FEATURE FRAMES (sherpa-onnx style:
# num_processed_frames / num_frames_ready / chunk_size).  The model emits
# one output row every `factor` input frames (kaldi anchors row k at
# frame k*factor -- nnet-compile-looped.cc), and the kaldi
# lattice decoder consumes one row at a time -- rows are formed only
# where they are fed to the Decoder, never padded or repeated.  To emit
# the rows of frames [t, end)
# the recognizer runs the AM on the window [t - L, last_anchor + R]
# (start snapped to the row grid) and feeds the rows straight through.

import json
import re
from pathlib import Path

import kaldi_decoder as kd
import kaldi_native_fbank as knf
import kaldi_native_io
import kaldifst
import kaldi_hmm_gmm as khg
import numpy as np

import onnxruntime as ort

# The feature pipeline works at int16 scale (vosk's convention): float
# samples in [-1, 1] are multiplied by this before the extractor.
INT16_SCALE = 32768.0


# ---------------------------------------------------------------------------
# feature-pipeline helpers
# ---------------------------------------------------------------------------


def read_meta(model_dir):
    """am/model_meta.json: dims, subsampling, sample rate, receptive field."""
    return json.loads((Path(model_dir) / "am" / "model_meta.json").read_text())


def load_audio(wav, sample_rate):
    """wav -> (mono float32 samples at `sample_rate`): librosa reads and
    resamples in one call."""
    import librosa

    return librosa.load(str(wav), sr=sample_rate)


def default_kaldi_src():
    """Locate kaldi's src/ for the reference tools ($KALDI_ROOT/src,
    /opt/kaldi/src on the docker image, or /Users/fangjun/open-source/kaldi/src)."""
    import os

    for cand in (
        os.environ.get("KALDI_ROOT"),
        "/opt/kaldi",  # the kaldi docker image
        "/Users/fangjun/open-source/kaldi",
    ):
        if cand and Path(cand, "src").is_dir():
            return str(Path(cand) / "src")
    return None


def run(cmd):
    """subprocess.run(check) but show the tool's stderr when it fails."""
    import subprocess

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        stderr_tail = r.stderr.decode(errors="replace")[-2000:]
        raise SystemExit(
            f"command failed ({r.returncode}): {cmd[0]}\n{stderr_tail}"
        )
    return r


def ceil_div(x, divisor):
    """How many `divisor`-sized rows are needed to hold x items."""
    return (x + divisor - 1) // divisor


def round_up(x, multiple):
    """Smallest multiple of `multiple` that is >= x (rounding to the
    model's row grid, which moves in steps of `multiple`)."""
    return ceil_div(x, multiple) * multiple


def read_model_conf(conf_path):
    """Parse conf/model.conf (kaldi ParseOptions --name=value lines)."""
    out = {}
    for line in Path(conf_path).read_text().splitlines():
        m = re.match(r"--([A-Za-z0-9._\-]+)=(\S+)$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def mfcc_options(conf_path):
    """knf.MfccOptions from conf/mfcc.conf; unknown keys are reported and
    ignored.  Dither is forced to 0 for determinism."""
    opts = knf.MfccOptions()
    opts.frame_opts.dither = 0
    for key, value in read_model_conf(conf_path).items():
        if key == "use-energy":
            opts.use_energy = value.lower() == "true"
        elif key == "sample-frequency":
            opts.frame_opts.samp_freq = float(value)
        elif key == "num-mel-bins":
            opts.mel_opts.num_bins = int(value)
        elif key == "num-ceps":
            opts.num_ceps = int(value)
        elif key == "low-freq":
            opts.mel_opts.low_freq = float(value)
        elif key == "high-freq":
            opts.mel_opts.high_freq = float(value)
        else:
            print(f"WARNING: ignoring mfcc.conf key {key}={value}")
    return opts


class IvectorEngine:
    """kaldi-hmm-gmm online i-vector extractor (ivector/ of the model).

    max_count: prior scaling threshold in frames (100 = the vosk runtime
    value; 0 disables it).  splice context comes from ivector/splice.conf;
    the rest keeps the khg constructor defaults.
    """

    def __init__(self, model_dir, max_count=100.0):
        iv = Path(model_dir) / "ivector"
        lda = kaldi_native_io.FloatMatrix.read(str(iv / "final.mat")).numpy()
        gs = kaldi_native_io.DoubleMatrix.read(str(iv / "global_cmvn.stats")).numpy()
        extractor = khg.IvectorExtractor.read(str(iv / "final.ie"))
        ubm = khg.read_diag_gmm(str(iv / "final.dubm"))
        splice = read_model_conf(iv / "splice.conf")
        cfg = khg.OnlineIvectorExtractorConfig(
            left_context=int(splice["left-context"]),
            right_context=int(splice["right-context"]),
            lda_mat=lda.astype(np.float32),
            global_cmvn_stats=gs,
            max_count=max_count,
        )
        self.engine = khg.OnlineIvectorExtractor(cfg, extractor, ubm)
        # the engine borrows extractor/ubm by C++ reference (config is
        # copied); this wrapper owns them for the engine's lifetime
        self._refs = (extractor, ubm)

    def accept(self, feats):
        """Feed new rows [T_new, F]; returns the running estimate [S]."""
        if len(feats):
            self.engine.accept_features(feats)
        return np.array(self.engine.latest_ivector, dtype=np.float32)

    def flush(self):
        """Mark end of input; returns the final estimate [S]."""
        self.engine.flush()
        return np.array(self.engine.latest_ivector, dtype=np.float32)


def load_graph(model_dir):
    """Small models: lazy HCLr o Gr lookahead composition (like vosk-api);
    large models: graph/HCLG.fst directly.  Never materialize the
    composition -- keep the inputs in their native FST types."""
    graph = Path(model_dir) / "graph"
    if (graph / "HCLG.fst").is_file():
        return kaldifst.StdFst.read(str(graph / "HCLG.fst"))
    hcl = kaldifst.StdFst.read(str(graph / "HCLr.fst"))
    gr = kaldifst.StdFst.read(str(graph / "Gr.fst"))
    disambig = [int(x) for x in (graph / "disambig_tid.int").read_text().split()]
    return kaldifst.lookahead_compose_fst(hcl, gr, disambig)


def load_words(model_dir, fst):
    """Word-id -> text (words.txt, else the graph's output symbols)."""
    path = Path(model_dir) / "graph" / "words.txt"
    if path.is_file():
        words = {}
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                words[int(parts[1])] = parts[0]  # text id
        return words
    syms = fst.output_symbols
    return {
        wid: syms.find(wid) for wid in range(1, syms.available_key()) if syms.find(wid)
    }


# ---------------------------------------------------------------------------
# Decoder -- kaldi FST lattice decoder, fed one model-output row per
# decoder frame (a row every `factor` input feature frames; rows are
# formed only at accept_loglikes)
# ---------------------------------------------------------------------------


# Per-model decoder resources, loaded once per process.  The lazy HCLr o Gr
# composition expands state-by-state (~2 s for small models) and caches the
# expansion on the FST object -- reloading the graph per utterance would pay
# that cost again every time.
_DECODER_RESOURCES = {}


def _decoder_resources(model_dir):
    key = str(Path(model_dir).resolve())
    if key not in _DECODER_RESOURCES:
        tm = khg.TransitionModel()
        tm.read(str(Path(model_dir) / "am" / "transition_model.mdl"))
        fst = load_graph(model_dir)
        _DECODER_RESOURCES[key] = (tm, fst, load_words(model_dir, fst))
    return _DECODER_RESOURCES[key]


class Decoder:
    """A kaldi LatticeFasterDecoder over the model's output rows.

    The kaldi decoder consumes ONE log-prob row per decoder frame; the
    model emits one row every `factor` input feature frames (kaldi
    anchors row k at input frame k*factor -- the same mapping
    `LogLikelihood(subsampled_frame, index)` uses, never repeating
    rows).  Everything outside this class counts input frames;
    rows are formed at the accept_loglikes() boundary below.

    Lifecycle (one utterance):
        accept_loglikes(rows)  -- feed new rows, advance the lattice once
                                  (feed each row exactly once; rows are
                                  scaled by `acoustic_scale`, like
                                  DecodableMatrixScaledMapped)
        ... more accept_loglikes calls as audio arrives ...
        finalize()             -- end of input: marks the decodable
                                  finished and finalizes the lattice
                                  (idempotent)
        get_text()             -- running 1-best; may REVISE earlier
                                  output as more evidence arrives (normal
                                  ASR behavior), stable only after
                                  finalize()

    The transition model / graph / words are shared per model dir (see
    _decoder_resources -- the lazy graph expansion is cached there);
    reset() starts a new utterance's lattice state on the same graph.
    """

    def __init__(self, model_dir, acoustic_scale):
        self.tm, self.fst, self.words = _decoder_resources(model_dir)
        self.acoustic_scale = acoustic_scale
        self.decode_config = self._decode_config(model_dir)
        self.reset()

    @staticmethod
    def _decode_config(model_dir):
        """LatticeFasterDecoderConfig aligned with the model.

        The model's conf/model.conf carries the decode params the vosk
        runtime uses at load time ("Decoding params beam=..." in its log):
        --beam / --max-active / --min-active / --lattice-beam.  Keys the
        conf does not set keep the kaldi library defaults (standard
        ParseOptions semantics -- defaults, not guesses)."""
        cfg = kd.LatticeFasterDecoderConfig()
        conf = read_model_conf(Path(model_dir) / "conf" / "model.conf")
        for key, attr, cast in (
            ("beam", "beam", float),
            ("max-active", "max_active", int),
            ("min-active", "min_active", int),
            ("lattice-beam", "lattice_beam", float),
        ):
            if key in conf:
                setattr(cfg, attr, cast(conf[key]))
        return cfg

    def reset(self):
        """Start a new utterance (model and graph stay loaded)."""
        self.decodable = kd.DecodableMatrixMappedOffset(self.tm)
        self.faster_decoder = kd.LatticeFasterDecoder(
            self.fst, self.decode_config
        )
        self.faster_decoder.init_decoding()
        self.finalized = False

    def accept_loglikes(self, rows):
        """Feed model-output rows [n, num_pdfs] -- one row per decoder
        frame, i.e. the rows of n/factor input feature frames -- and
        advance once (each row is fed exactly once)."""
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if self.acoustic_scale != 1.0:
            rows = rows * self.acoustic_scale
        self.decodable.accept_loglikes(rows)
        self.faster_decoder.advance_decoding(self.decodable)

    def finalize(self):
        """Mark end of input and finish decoding (idempotent)."""
        if self.finalized:
            return
        self.decodable.input_is_finished()
        self.faster_decoder.finalize_decoding()
        self.finalized = True

    def get_text(self):
        """Current 1-best (may revise as more audio arrives -- normal ASR
        behavior)."""
        if self.faster_decoder.num_frames_decoded() == 0:
            return ""  # nothing fed yet (e.g. the first small chunk)
        ok, lat = self.faster_decoder.get_best_path()
        if not ok:
            return ""
        _, _, word_ids, _ = kd.lattice_best_path(lat)
        return " ".join(self.words.get(w, f"<{w}>") for w in word_ids if w != 0)


# ---------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------


class OnlineStream:
    """One utterance of streaming audio: samples -> features + i-vector.

    Like sherpa-onnx's OnlineStream (online-stream.h): AcceptWaveform /
    InputFinished / NumFramesReady / GetFrames.  The stream is the
    feature pipeline AND the per-utterance decode state (the C++ design
    splits these across OnlineStream and OnlineStreamState):

        mfcc, engine   -- MFCC extractor + running i-vector; the features
                          live in the extractor (num_frames_ready /
                          get_frames read them, nothing is cached) and
                          the i-vector estimate updates inside
                          accept_waveform() as rows become ready
        ivector        -- current i-vector estimate [S] (None until the
                          first frames arrive; final after
                          input_finished())
        finished       -- set by input_finished(); enables the flush rule
                          in OnlineRecognizer.is_ready
        decoder        -- this utterance's Decoder
        num_processed_frames -- input feature frames PROCESSED: the
                          model rows anchored before this position have
                          been computed and fed to `decoder` (sharpa-onnx
                          naming: GetNumProcessedFrames(); the recognizer
                          advances it by chunk_size each decode step).
                          The model's row count is
                          num_processed_frames // factor.

    accept_waveform() takes the sample rate and checks it against the
    model's (like sherpa-onnx's AcceptWaveform(sampling_rate, waveform,
    n); the drivers resample with librosa before feeding).
    """

    def __init__(self, model_dir, mfcc_opts, max_count=100.0,
                 acoustic_scale=1.0):
        self.model_dir = Path(model_dir)
        self.mfcc_opts = mfcc_opts
        self.max_count = max_count
        self.sample_rate = mfcc_opts.frame_opts.samp_freq
        # per-stream decode state (sharpa-onnx's OnlineStreamState)
        self.decoder = Decoder(self.model_dir, acoustic_scale)
        self.reset()

    def reset(self):
        """Start a new utterance (sharpa-onnx's OnlineStream::Reset; the
        stream object, its decoder and the loaded model stay)."""
        self.mfcc = knf.OnlineMfcc(self.mfcc_opts)
        self.engine = IvectorEngine(self.model_dir, self.max_count)
        self.ivector = None
        self.finished = False
        self._iv_frames = 0  # frames already fed to the i-vector
        # input frames processed (the model rows anchored before this
        # position are computed and fed to the decoder)
        self.num_processed_frames = 0
        self.decoder.reset()

    def accept_waveform(self, sample_rate, samples):
        """samples: [N] float32 at `sample_rate`; the rate must match the
        model's (sherpa-onnx's AcceptWaveform(sampling_rate, waveform, n)
        resamples internally -- here we only check; the drivers resample
        with librosa)."""
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"sample rate {sample_rate} != model rate {self.sample_rate}; "
                f"resample first (the drivers use librosa.load(..., sr=...))"
            )
        # new feature rows and the running i-vector update as they
        # become ready
        self.mfcc.accept_waveform(self.sample_rate, (samples * INT16_SCALE).tolist())
        self._drain()

    def input_finished(self):
        """No more samples; flush the tail features and the i-vector."""
        self.mfcc.input_finished()
        self._drain()
        self.ivector = self.engine.flush()
        self.finished = True

    @property
    def num_frames_ready(self):
        return self.mfcc.num_frames_ready

    def get_frames(self, start, n):
        """[n, F] feature rows starting at row `start` (the AM window).

        Read straight from the extractor (nothing is cached); frames
        outside the stream are replicated at the edge -- kaldi's edge
        rule -- so callers can ask for any window without clamping."""
        last = self.num_frames_ready - 1
        return np.stack([
            np.array(self.mfcc.get_frame(min(max(i, 0), last)), dtype=np.float32)
            for i in range(start, start + n)
        ])

    def is_last_frame(self, i):
        return self.finished and i == self.num_frames_ready - 1

    def _drain(self):
        """Update the running i-vector with any new feature rows."""
        n = self.num_frames_ready
        if n > self._iv_frames:
            self.ivector = self.engine.accept(
                np.stack([
                    np.array(self.mfcc.get_frame(i), dtype=np.float32)
                    for i in range(self._iv_frames, n)
                ])
            )
            self._iv_frames = n


class OfflineStream:
    """One utterance of audio: samples -> features + i-vector (whole
    utterance at once), like sherpa-onnx's OfflineStream (offline-stream.h)
    with AcceptWaveform / InputFinished / GetFrames.

    accept_waveform() only accumulates samples; input_finished() does the
    per-utterance frontend work in one go (MFCC + one whole-utterance
    i-vector estimate) and freezes the features/i-vector for the recognizer.
    """

    def __init__(self, model_dir, mfcc_opts, max_count=100.0,
                 acoustic_scale=1.0):
        self.model_dir = Path(model_dir)
        self.mfcc_opts = mfcc_opts
        self.sample_rate = mfcc_opts.frame_opts.samp_freq
        self.max_count = max_count
        self._samples = []
        self.mfcc = None
        self.ivector = None
        # per-stream decode state (sharpa-onnx's OnlineStreamState)
        self.decoder = Decoder(self.model_dir, acoustic_scale)

    def accept_waveform(self, sample_rate, samples):
        """samples: [N] float32 at `sample_rate`; the rate must match the
        model's (sherpa-onnx's AcceptWaveform(sampling_rate, waveform, n)
        resamples internally -- here we only check; the drivers resample
        with librosa)."""
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"sample rate {sample_rate} != model rate {self.sample_rate}; "
                f"resample first (the drivers use librosa.load(..., sr=...))"
            )
        self._samples.append(np.asarray(samples, dtype=np.float32))

    def input_finished(self):
        samples = np.concatenate(self._samples) if self._samples else np.zeros(0)
        self.mfcc = knf.OnlineMfcc(self.mfcc_opts)
        self.mfcc.accept_waveform(self.sample_rate, (samples * INT16_SCALE).tolist())
        self.mfcc.input_finished()
        engine = IvectorEngine(self.model_dir, self.max_count)
        engine.accept(self.get_frames(0, self.num_frames_ready))
        self.ivector = engine.flush()

    @property
    def num_frames_ready(self):
        return 0 if self.mfcc is None else self.mfcc.num_frames_ready

    def get_frames(self, start, n):
        """[n, F] feature rows starting at row `start` (the AM window),
        with the same edge rule as OnlineStream.get_frames: rows outside
        the utterance are replicated at the edge (kaldi's edge rule)."""
        last = self.num_frames_ready - 1
        return np.stack([
            np.array(self.mfcc.get_frame(min(max(i, 0), last)), dtype=np.float32)
            for i in range(start, start + n)
        ])


# ---------------------------------------------------------------------------
# AM backends -- the recognizers below only see .meta and .run(); the model
# (ONNX export / torch Nnet3Torch / generated FlatAm) is the only difference
# ---------------------------------------------------------------------------


class OnnxAm:
    """The exported ONNX AM: am/final.onnx (or am/final.int8.onnx).

    The graph emits one row per `factor` input frames directly (the
    rows ::factor are computed on that grid, see
    torch_model.DecimationPlan).
    """

    def __init__(self, model_dir, use_int8=False):
        self.model_dir = Path(model_dir)
        self.net = ort.InferenceSession(
            str(
                self.model_dir
                / "am"
                / ("final.int8.onnx" if use_int8 else "final.onnx")
            ),
            providers=["CPUExecutionProvider"],
        )
        raw = self.net.get_modelmeta().custom_metadata_map
        self.meta = {
            "sample_rate": int(raw["sample_rate"]),
            "frame_subsampling_factor": int(raw["frame_subsampling_factor"]),
            "acoustic_scale": float(raw["acoustic_scale"]),
            "left_context": int(raw["left_context"]),
            "right_context": int(raw["right_context"]),
        }

    def run(self, feats, ivector):
        """feats [T, F] x ivector [S] -> log-probs
        [ceil(T/factor), num_pdfs]: one row per `factor` input
        frames."""
        return self.net.run(
            ["log_probs"],
            {
                "features": feats[None],
                "ivector": np.ascontiguousarray(ivector)[None],
            },
        )[0][0]


class TorchAm:
    """The torch Nnet3Torch AM (torch model dir: am/am.pt).

    The net computes the full 10 ms grid; run() keeps the rows
    ::factor (row k is anchored at input frame k*factor).
    """

    def __init__(self, model_dir):
        import torch

        from torch_model import Nnet3Torch

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir)
        payload = torch.load(
            self.model_dir / "am" / "am.pt", map_location="cpu", weights_only=False
        )
        self.net = Nnet3Torch.from_export_args(payload["args"])
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()

    def run(self, feats, ivector):
        torch = self._torch
        with torch.no_grad():
            full = self.net(
                torch.from_numpy(feats)[None], torch.from_numpy(ivector)[None]
            )[0].numpy()
        return full[:: self.meta["frame_subsampling_factor"]]


class FlatAm:
    """The generated explicit-wiring module (torch model dir:
    am/am_flat.py + am/am.pt).  Same contract as TorchAm."""

    def __init__(self, model_dir):
        import importlib.util

        import torch

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir)
        spec = importlib.util.spec_from_file_location(
            "am_flat", self.model_dir / "am" / "am_flat.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.net = mod.FlatAm.from_am_pt(str(self.model_dir / "am" / "am.pt"))

    def run(self, feats, ivector):
        torch = self._torch
        with torch.no_grad():
            full = self.net(
                torch.from_numpy(feats)[None], torch.from_numpy(ivector)[None]
            )[0].detach().numpy()
        return full[:: self.meta["frame_subsampling_factor"]]


# ---------------------------------------------------------------------------
# recognizers
# ---------------------------------------------------------------------------


class _RecognizerBase:
    """Shared plumbing: the AM backend + the model's subsampling contract.

    The backend (OnnxAm / TorchAm / FlatAm) supplies
      * `meta`: sample_rate, frame_subsampling_factor,
        left_context, right_context, acoustic_scale (from the model's
        model_meta.json or the ONNX embedded metadata -- never assumed);
      * `run(feats, ivector)`: [T, F] features x [S] i-vector ->
        log-probs [ceil(T/factor), num_pdfs] -- one row per
        `factor` input feature frames (row k at frame k*factor).
    Everything above the backend -- streams, windowing, the Decoder --
    is model-agnostic.
    """

    def __init__(self, am, max_count=100.0):
        self.am = am
        self.model_dir = am.model_dir
        self.max_count = max_count
        self.mfcc_opts = mfcc_options(self.model_dir / "conf" / "mfcc.conf")
        self.meta = am.meta
        self.sample_rate = self.meta["sample_rate"]
        self.factor = self.meta["frame_subsampling_factor"]
        # the model's receptive field, sherpa-onnx naming
        # (online-recognizer-*-impl.h): left_context / right_context
        self.left_context = self.meta["left_context"]
        self.right_context = self.meta["right_context"]
        # left context rounded up to the row grid (rows are every
        # `factor` frames): decode windows then start on an anchor
        self._grid_left = round_up(self.left_context, self.factor)

    def _run_step(self, stream, t, rows):
        """Run the model for one step and return exactly the step's rows:
        [rows, num_pdfs], anchored at input frames t, t+factor, ...

            window (t - L, t + rows*factor + R)  ->  [t, t + rows*factor)
            rows  ->  subsampled ->  (rows, num_pdfs)

        The model is stateless and PURE: the window carries the L + R
        context (it must be *computed with*, it is not *returned*) and
        the edges of the stream are replicated there (kaldi's edge rule).
        The model emits one row per `factor` frames of the whole window;
        the rows anchored at [t, t + rows*factor) are the step's own.
        (Subsampling lives inside the graph; crop-then-subsample ==
        subsample-then-crop.)"""
        window = stream.get_frames(
            t - self._grid_left,
            self._grid_left + rows * self.factor + self.right_context,
        )
        window_rows = self.am.run(window, stream.ivector)
        first_row = self._grid_left // self.factor
        return window_rows[first_row : first_row + rows]


class OnlineRecognizer(_RecognizerBase):
    """Streaming ASR, sherpa-onnx OnlineRecognizer style.

    create_stream() -> one utterance's OnlineStream; feed samples to the
    stream, then `while rec.is_ready(stream): rec.decode_stream(stream)`
    and read rec.get_result(stream) -- the hypothesis MAY REVISE earlier
    output as more evidence arrives (normal ASR behavior).

    One decode step = one AM call + one lattice advance; no row is ever
    scored twice.  `chunk_size` is the model's chunk -- sherpa-onnx's
    ChunkSize() (online-recognizer-*-impl.h, same variable names): every
    step runs the model on `L + chunk_size + R` frames and keeps the
    middle `chunk_size` frames' rows; stream.num_processed_frames
    advances by chunk_size each step.
    """

    def __init__(self, am, chunk_size, max_count=100.0):
        super().__init__(am, max_count)
        if chunk_size % self.factor:
            raise ValueError(
                f"chunk_size={chunk_size} must be a multiple of "
                f"frame_subsampling_factor={self.factor} (rows anchor "
                f"every {self.factor} x 10 ms)"
            )
        self.chunk_size = chunk_size  # input frames per decode step

    def create_stream(self):
        return OnlineStream(
            self.model_dir, self.mfcc_opts,
            max_count=self.max_count,
            acoustic_scale=self.meta["acoustic_scale"],
        )

    def is_ready(self, stream):
        """Is this stream ready for another decode step?

        All bookkeeping counts INPUT FEATURE FRAMES, like
        sherpa-onnx's online-recognizer-*-impl.h (same variable names):

            num_processed_frames += chunk_size       # per decode step

        The sherpa rule

            ready <=> num_frames_ready >
                      num_processed_frames + chunk_size + L + R

        relaxed on the left to

            num_frames_ready > num_processed_frames + chunk_size + R

        because the left L frames never need to be *waited* for: they
        come from the stream's history, or -- at the very start of the
        stream -- from kaldi-style replicated-edge padding inside the
        model, exactly as a one-shot run sees its left edge.  (Waiting
        for L too would only delay the first result.)

        Timeline (factor=3, L=26, R=14; one output row per `factor`
        frames, anchored at 10 ms frame k*factor):

            t = num_processed_frames (input frames, a multiple of
            chunk_size):
            window    = [t - L, t + chunk_size + R)   (L+chunk+R frames)
            kept rows = the model's rows anchored in [t, t + chunk_size)

        is_ready() is True when

          1. the frames for a whole step plus right context have
             arrived, OR
          2. the stream has finished (input_finished()) and something is
             still owed: the final SHORT chunk (fewer frames than a step
             -- like the C++ "is_final: accept short chunks" handling)
             or the decoder's end-of-input bookkeeping
             (`decoder.finalize()`, done by decode_stream).
        """
        t = stream.num_processed_frames
        if stream.num_frames_ready > t + self.chunk_size + self.right_context:
            return True
        # case 2: end-of-input remainder / finalize
        return stream.finished and (
            stream.num_frames_ready > t or not stream.decoder.finalized
        )

    def decode_stream(self, stream):
        """Decode one step (sharpa-onnx's DecodeStream): run the model on
        the L + chunk_size + R window at the current position and feed
        its rows to the decoder.  Each row is fed exactly once; the final
        (short) chunk at end of input goes out edge-clipped like a
        one-shot run, and the decoder is finalized."""
        factor = self.factor
        t = stream.num_processed_frames
        # rows for this step: a full chunk, or what is left at end of
        # input (a short final chunk may take several steps to drain)
        remaining_rows = ceil_div(stream.num_frames_ready - t, factor)
        rows = min(self.chunk_size // factor, remaining_rows)
        if rows:
            step_rows = self._run_step(stream, t, rows)
            stream.decoder.accept_loglikes(step_rows)
        stream.num_processed_frames += self.chunk_size
        if stream.finished and stream.num_processed_frames >= stream.num_frames_ready:
            stream.decoder.finalize()  # tail rows: right context clipped

    def get_result(self, stream):
        """Current 1-best hypothesis (may REVISE as more audio arrives).
        The lattice's best path is recomputed on every call."""
        return stream.decoder.get_text()

    def reset(self, stream):
        """Start a new utterance on this stream (sharpa-onnx's
        Reset(stream); model and graph stay loaded)."""
        stream.reset()


class OfflineRecognizer(_RecognizerBase):
    """Whole-utterance ASR, sherpa-onnx OfflineRecognizer style:

        rec = OfflineRecognizer(OnnxAm(model_dir))
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()
        rec.decode_stream(stream)
        print(rec.get_result(stream))

    There is no is_ready step: the whole utterance goes through the AM in
    one call (the graph handles any T), and its output rows feed
    the decoder directly.  This is the reference for the streaming path --
    both decode to the same text.
    """

    def create_stream(self):
        return OfflineStream(
            self.model_dir, self.mfcc_opts,
            max_count=self.max_count,
            acoustic_scale=self.meta["acoustic_scale"],
        )

    def decode_stream(self, stream):
        """Run the AM over the whole utterance in one step (sharpa-onnx's
        DecodeStream): the step spans every row of the stream, so the
        same window/crop convention as streaming applies."""
        rows = ceil_div(stream.num_frames_ready, self.factor)
        step_rows = self._run_step(stream, 0, rows)
        stream.decoder.accept_loglikes(step_rows)
        stream.decoder.finalize()

    def get_result(self, stream):
        return stream.decoder.get_text()
