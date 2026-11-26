#!/usr/bin/env python3
# sherpa-onnx/scripts/vosk/recognizer.py
#
# Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
#
# Python recognizer for the exported vosk models.  It reproduces the vosk
# runtime (vosk-api src/recognizer.cc, built on kaldi's online2) step by
# step, so that the ONNX model decodes to the SAME text as the vosk package:
#
#   vosk / kaldi online2                       here
#   ----------------------------------------   -------------------------------
#   OnlineNnet2FeaturePipeline: MFCC           knf.OnlineMfcc (dither 0)
#   OnlineIvectorFeature (kaldi defaults,      khg.OnlineIvectorFeature
#     max_count 100, use-most-recent)
#   OnlineSilenceWeighting (weight 1e-3,       khg.OnlineSilenceWeighting
#     --endpoint.silence-phones)
#   DecodableNnetLoopedOnline: chunks of       IvectorChunker + the AM backend
#     frames_per_chunk (24) frames, one          (OnnxAm / TorchAm / FlatAm):
#     i-vector per chunk, ivector input          a pure model fed the chunk's
#     rewritten to Round(ivector, chunk)         window with a per-frame
#                                                ivector [T, S]
#   SingleUtteranceNnet3IncrementalDecoder     kd.LatticeFasterOnlineDecoder
#   AcceptWaveform: 0.2 s steps, each:         OnlineStream.accept_waveform
#     UpdateSilenceWeights(); AdvanceDecoding()  (same steps, same order)
#   FinalResult: InputFinished,                OnlineStream.input_finished
#     UpdateSilenceWeights, AdvanceDecoding,
#     FinalizeDecoding
#   GetResult with SetMaxAlternatives(1):      lattice best path with the
#     graph cost x 0.9, 1-best                   graph cost x 0.9
#
# Verified on vosk-model-small-ko-0.22 against an instrumented libvosk:
# per-chunk i-vectors within 2e-5, silence delta-weights identical,
# acoustic rows within 1e-4, text identical to vosk's 1-best (with dither
# off in vosk -- vosk's default MFCC dither is random; see
# compare_onnx.vosk_baseline).  vosk's default result (max_alternatives=0)
# is an MBR decoding of the same lattice; it is not reproduced.  Neither
# is endpointing (AcceptWaveform's return value).
#
# Usage (the loop of sherpa-onnx's C++ microphone example):
#
#     rec = OnlineRecognizer(OnnxAm(model_dir))
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
# OfflineRecognizer runs the same pipeline on a whole utterance (vosk has
# no separate offline mode: a whole-file AcceptWaveform + FinalResult is
# this streaming pipeline in 0.2 s steps).

import json
import math
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

# vosk's Recognizer::AcceptWaveform processes its input in steps of 0.2 s:
# after each step it updates the silence weights and advances the decoder.
VOSK_FEED_SECONDS = 0.2

# vosk's Model::ReadDataFiles: silence_weighting_config.silence_weight
VOSK_SILENCE_WEIGHT = 1e-3

# vosk's Model::ReadDataFiles: ivector_extraction_opts.max_count
VOSK_IVECTOR_MAX_COUNT = 100.0

# vosk's Recognizer::GetResult: fst::ScaleLattice(GraphLatticeScale(0.9))
VOSK_LM_SCALE = 0.9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def to_int16(samples):
    """float samples in [-1, 1] -> int16 PCM (clipped, truncated toward
    zero).  vosk takes int16 audio; the feature pipeline sees these exact
    values (compare_onnx.vosk_baseline hands vosk the same int16 data)."""
    x = np.asarray(samples, dtype=np.float32) * INT16_SCALE
    return np.clip(x, -32768, 32767).astype(np.int16)


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
        raise SystemExit(f"command failed ({r.returncode}): {cmd[0]}\n{stderr_tail}")
    return r


def ceil_div(x, divisor):
    """How many `divisor`-sized rows are needed to hold x items."""
    return (x + divisor - 1) // divisor


def round_up(x, multiple):
    """Smallest multiple of `multiple` that is >= x (rounding to the
    model's row grid, which moves in steps of `multiple`)."""
    return ceil_div(x, multiple) * multiple


def read_model_conf(conf_path):
    """Parse a kaldi config file (--name=value lines)."""
    out = {}
    for line in Path(conf_path).read_text().splitlines():
        # kaldi's ParseOptions strips inline '#' comments before parsing
        # (parse-options.cc ReadConfigFile) -- do the same, or a line like
        # '--sample-frequency=8000 # telephone' is dropped entirely
        line = line.split("#", 1)[0].strip()
        m = re.match(r"--([A-Za-z0-9._\-]+)=(\S+)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def kaldi_bool(value):
    return value.lower() in ("true", "1", "t")


def mfcc_options(conf_path):
    """knf.MfccOptions from conf/mfcc.conf; unknown keys are reported and
    ignored.  Dither is forced to 0 for determinism."""
    opts = knf.MfccOptions()
    opts.frame_opts.dither = 0
    for key, value in read_model_conf(conf_path).items():
        if key == "use-energy":
            opts.use_energy = kaldi_bool(value)
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
        elif key in ("allow-upsample", "allow-downsample", "dither"):
            pass  # resampling is done by the caller; dither is always 0
        else:
            print(f"WARNING: ignoring mfcc.conf key {key}={value}")
    return opts


def kaldi_chunk_size(frames_per_chunk, frame_subsampling_factor, modulus):
    """kaldi's GetChunkSize (nnet-compile-looped.cc): the smallest
    chunk >= frames_per_chunk that is a multiple of the network's modulus
    and of the subsampling factor."""
    chunk = frames_per_chunk
    while chunk % modulus or chunk % frame_subsampling_factor:
        chunk += 1
    return chunk


# ---------------------------------------------------------------------------
# the vosk configuration of a model directory
# ---------------------------------------------------------------------------


class VoskConfig:
    """What the vosk runtime reads from a model directory (vosk-api
    src/model.cc ConfigureV2 + ReadDataFiles), with kaldi's defaults for
    every option conf/model.conf does not set.

    conf/model.conf: decoder (--beam --max-active --min-active
      --lattice-beam ...), looped decodable (--frames-per-chunk
      --acoustic-scale --frame-subsampling-factor), endpointing
      (--endpoint.silence-phones: also the silence phones of the i-vector
      silence weighting)
    ivector/: splice.conf, online_cmvn.conf, final.mat, global_cmvn.stats,
      final.dubm, final.ie
    """

    def __init__(self, model_dir, modulus=1):
        model_dir = Path(model_dir)
        conf = read_model_conf(model_dir / "conf" / "model.conf")
        self.conf = conf

        # LatticeIncrementalDecoderConfig (vosk's decoder) defaults
        dcfg = kd.LatticeFasterDecoderConfig()
        dcfg.prune_scale = 0.01  # the incremental decoder's default
        for key, attr, cast in (
            ("beam", "beam", float),
            ("max-active", "max_active", int),
            ("min-active", "min_active", int),
            ("lattice-beam", "lattice_beam", float),
            ("prune-interval", "prune_interval", int),
            ("beam-delta", "beam_delta", float),
            ("hash-ratio", "hash_ratio", float),
            ("prune-scale", "prune_scale", float),
        ):
            if key in conf:
                setattr(dcfg, attr, cast(conf[key]))
        self.decoder_config = dcfg

        # NnetSimpleLoopedComputationOptions defaults
        self.frame_subsampling_factor = int(conf.get("frame-subsampling-factor", "1"))
        self.acoustic_scale = float(conf.get("acoustic-scale", "0.1"))
        self.frames_per_chunk = kaldi_chunk_size(
            int(conf.get("frames-per-chunk", "24")),
            self.frame_subsampling_factor,
            modulus,
        )
        if int(conf.get("extra-left-context-initial", "0")):
            # only changes the output of recurrent nets (more blank-state
            # warm-up before the first chunk)
            print(
                "WARNING: --extra-left-context-initial is not supported; "
                "ignored (no effect on TDNN models)"
            )

        # OnlineSilenceWeightingConfig as vosk sets it
        self.silence_config = khg.OnlineSilenceWeightingConfig(
            silence_phones_str=conf.get("endpoint.silence-phones", ""),
            silence_weight=VOSK_SILENCE_WEIGHT,
        )

        # OnlineIvectorExtractionConfig: kaldi defaults + vosk's max_count
        iv = model_dir / "ivector"
        splice = read_model_conf(iv / "splice.conf")
        icfg = khg.OnlineIvectorExtractorConfig(
            left_context=int(splice.get("left-context", "4")),
            right_context=int(splice.get("right-context", "4")),
            lda_mat=kaldi_native_io.FloatMatrix.read(str(iv / "final.mat")).numpy(),
            global_cmvn_stats=kaldi_native_io.DoubleMatrix.read(
                str(iv / "global_cmvn.stats")
            ).numpy(),
            max_count=VOSK_IVECTOR_MAX_COUNT,
        )
        cmvn_conf = iv / "online_cmvn.conf"
        if cmvn_conf.is_file():
            for key, attr, cast in (
                ("cmn-window", "cmn_window", int),
                ("speaker-frames", "speaker_frames", int),
                ("global-frames", "global_frames", int),
                ("norm-means", "normalize_mean", kaldi_bool),
                ("norm-vars", "normalize_variance", kaldi_bool),
                ("skip-dims", "skip_dims", str),
            ):
                value = read_model_conf(cmvn_conf).get(key)
                if value is not None:
                    setattr(icfg, attr, cast(value))
        self.ivector_config = icfg
        self.ivector_extractor = khg.IvectorExtractor.read(str(iv / "final.ie"))
        self.ubm = khg.read_diag_gmm(str(iv / "final.dubm"))


# ---------------------------------------------------------------------------
# decoding graph (shared per model dir)
# ---------------------------------------------------------------------------


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


# Per-model resources, loaded once per process.  The lazy HCLr o Gr
# composition expands state-by-state (~2 s for small models) and caches the
# expansion on the FST object -- reloading the graph per utterance would pay
# that cost again every time.
_RESOURCES = {}


def _model_resources(model_dir):
    key = str(Path(model_dir).resolve())
    if key not in _RESOURCES:
        tm = khg.TransitionModel()
        tm.read(str(Path(model_dir) / "am" / "transition_model.mdl"))
        fst = load_graph(model_dir)
        _RESOURCES[key] = (tm, fst, load_words(model_dir, fst))
    return _RESOURCES[key]


# ---------------------------------------------------------------------------
# AM backends -- the recognizer only sees .meta, .modulus and .run(); the
# model (ONNX export / torch Nnet3Torch / generated FlatAm) is the only
# difference.
#
#   run(feats [T, F], ivector [T, S], state, shift) -> (rows, state_out)
#
# The ivector is per input frame (the chunk's i-vector repeated over the
# chunk's frames, see IvectorChunker).  `state` is the recurrent state
# carried between windows (models with an LSTM; None = blank); `shift` is
# the number of new input frames of the window (the next window starts
# `shift` frames later).  Models without recurrence ignore both and return
# state_out = None.
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
            "left_context": int(raw["left_context"]),
            "right_context": int(raw["right_context"]),
        }
        if "modulus" not in raw:
            raise ValueError(
                f"{self.model_dir}: the ONNX model has no 'modulus' metadata "
                "(exported before the per-frame ivector input); re-export it "
                "with export_onnx.py"
            )
        self.modulus = int(raw["modulus"])
        # recurrent models take/return the state [1, state_rows, state_dim]
        self.state_shape = None
        if int(raw.get("recurrent", 0)):
            self.state_shape = (1, int(raw["state_rows"]), int(raw["state_dim"]))

    def run(self, feats, ivector, state=None, shift=None):
        """feats [T, F] x ivector [T, S] -> (log-probs [ceil(T/factor),
        num_pdfs], state_out): one row per `factor` input frames.  The
        graph derives `shift` from T (the window contract), so it is not
        an input."""
        feed = {
            "features": feats[None],
            "ivector": np.ascontiguousarray(ivector)[None],
        }
        if self.state_shape is None:
            return self.net.run(["log_probs"], feed)[0][0], None
        feed["state"] = (
            np.zeros(self.state_shape, dtype=np.float32) if state is None else state
        )
        log_probs, state_out = self.net.run(["log_probs", "state_out"], feed)
        return log_probs[0], state_out


class TorchAm:
    """The torch Nnet3Torch AM (torch model dir: am/am.pt).

    The net computes the full 10 ms grid; run() keeps the rows
    ::factor (row k is anchored at input frame k*factor).
    """

    def __init__(self, model_dir):
        import torch

        from torch_model import Nnet3Torch, nnet_modulus

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir)
        payload = torch.load(
            self.model_dir / "am" / "am.pt", map_location="cpu", weights_only=False
        )
        self.net = Nnet3Torch.from_export_args(payload["args"])
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        self.modulus = nnet_modulus(self.net.spec.nodes)

    def run(self, feats, ivector, state=None, shift=None):
        return _run_torch(
            self._torch, self.net, self.meta, feats, ivector, state, shift
        )


class FlatAm:
    """The generated explicit-wiring module (torch model dir:
    am/am_flat.py + am/am.pt).  Same contract as TorchAm."""

    def __init__(self, model_dir):
        import importlib.util

        import torch

        from torch_model import Nnet3Torch, nnet_modulus

        self._torch = torch
        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir)
        spec = importlib.util.spec_from_file_location(
            "am_flat", self.model_dir / "am" / "am_flat.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.net = mod.FlatAm.from_am_pt(str(self.model_dir / "am" / "am.pt"))
        payload = torch.load(
            self.model_dir / "am" / "am.pt", map_location="cpu", weights_only=False
        )
        self.modulus = nnet_modulus(
            Nnet3Torch.from_export_args(payload["args"]).spec.nodes
        )

    def run(self, feats, ivector, state=None, shift=None):
        return _run_torch(
            self._torch, self.net, self.meta, feats, ivector, state, shift
        )


def _run_torch(torch, net, meta, feats, ivector, state, shift):
    """TorchAm / FlatAm.run: full-rate net, rows ::factor kept; a
    recurrent net (state_rows > 0) runs forward_streaming with the
    carried state."""
    x = torch.from_numpy(feats)[None]
    iv = torch.from_numpy(np.ascontiguousarray(ivector))[None]
    factor = meta["frame_subsampling_factor"]
    with torch.no_grad():
        if not net.state_rows:
            return net(x, iv)[0].detach().numpy()[::factor], None
        if state is None:
            state = net.initial_state(1)
        full, state_out = net.forward_streaming(x, iv, state, shift)
    return full[0].detach().numpy()[::factor], state_out


# ---------------------------------------------------------------------------
# kaldi's looped decodable: chunks and their i-vectors
# ---------------------------------------------------------------------------


class IvectorChunker:
    """Which i-vector each input frame gets -- kaldi's
    DecodableNnetLoopedOnline (decodable-online-looped.cc).

    Chunk c (frames_per_chunk = N, left/right context L/R) reads the input
    frames [-L, N + R) (c = 0) or [c*N + R, (c+1)*N + R) (c > 0) -- each
    chunk the frames its predecessors did not -- and gets ONE i-vector,
    taken when the chunk is computed.  The network's
    ReplaceIndex(ivector, t, 0) is rewritten to Round(ivector, N)
    (ModifyNnetIvectorPeriod), and the computation requests ask for each
    i-vector time Round(t, N) once, in the first chunk that needs it.  So
    input frame t uses the i-vector of the chunk that holds frame
    Round(t, N) (or of chunk 0 if that frame is before the first chunk)."""

    def __init__(self, frames_per_chunk, right_context):
        self.chunk = frames_per_chunk
        self.right_context = right_context
        self.ivectors = []  # i-vector of chunk c

    def chunk_of(self, frame):
        """The chunk whose input range holds `frame`."""
        if frame < self.chunk + self.right_context:
            return 0
        return (frame - self.right_context) // self.chunk

    def rows(self, start, n):
        """[n, S] i-vectors of input frames [start, start + n)."""
        out = np.empty((n, self.ivectors[0].shape[0]), dtype=np.float32)
        for i, t in enumerate(range(start, start + n)):
            block = (t // self.chunk) * self.chunk
            out[i] = self.ivectors[self.chunk_of(block)]
        return out


# ---------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------


class OnlineStream:
    """One utterance, as vosk's Recognizer processes it.

    accept_waveform() queues the samples in vosk's 0.2 s steps (the steps
    are per call, like vosk's AcceptWaveform); the recognizer's
    decode_stream() processes one step: MFCC -> i-vector feature, silence
    weights from the current best path, then every chunk the looped
    decodable has ready is computed and decoded.  input_finished() queues
    vosk's FinalResult steps (flush, last silence update, last chunks,
    finalize).
    """

    def __init__(self, rec):
        self.rec = rec
        self.sample_rate = rec.sample_rate
        tm, self.fst, self.words = _model_resources(rec.model_dir)
        self.tm = tm
        self.reset()

    def reset(self):
        rec = self.rec
        cfg = rec.config
        self.mfcc = knf.OnlineMfcc(rec.mfcc_opts)
        self.ivector = khg.OnlineIvectorFeature(
            cfg.ivector_config, cfg.ivector_extractor, cfg.ubm
        )
        self.silence = khg.OnlineSilenceWeighting(
            self.tm, cfg.silence_config, rec.factor
        )
        self.decoder = kd.LatticeFasterOnlineDecoder(self.fst, cfg.decoder_config)
        self.decoder.init_decoding()
        self.decodable = kd.DecodableMatrixMappedOffset(self.tm)
        self.chunker = IvectorChunker(rec.chunk_size, rec.right_context)
        self.am_state = None  # recurrent AMs: blank state at the start
        self.num_chunks = 0  # chunks computed
        self.num_rows_decoded = 0  # decoder frames (model rows) fed
        self.pending = []  # queued 0.2 s steps (sample arrays)
        self.finished = False  # input_finished() called
        self.flushed = False  # the final step was processed
        self.finalized = False
        self._iv_fed = 0  # MFCC frames given to the i-vector feature

    def accept_waveform(self, sample_rate, samples):
        """samples: [N] float32 in [-1, 1] at `sample_rate` (must be the
        model's; resample first, e.g. librosa.load(..., sr=...))."""
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"sample rate {sample_rate} != model rate {self.sample_rate}; "
                f"resample first (the drivers use librosa.load(..., sr=...))"
            )
        assert not self.finished, "accept_waveform() after input_finished()"
        step = int(self.sample_rate * VOSK_FEED_SECONDS)
        samples = np.asarray(samples, dtype=np.float32)
        for i in range(0, len(samples), step):
            self.pending.append(samples[i : i + step])

    def input_finished(self):
        self.finished = True

    @property
    def num_frames_ready(self):
        """MFCC frames computed so far."""
        return self.mfcc.num_frames_ready

    @property
    def num_processed_frames(self):
        """Input frames whose model rows have been decoded."""
        return self.num_rows_decoded * self.rec.factor


class OfflineStream:
    """One whole utterance; decoded by OfflineRecognizer with the same
    pipeline as streaming (see OfflineRecognizer)."""

    def __init__(self, rec):
        self.sample_rate = rec.sample_rate
        self._samples = []
        self.online = OnlineStream(rec)

    def accept_waveform(self, sample_rate, samples):
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"sample rate {sample_rate} != model rate {self.sample_rate}; "
                f"resample first (the drivers use librosa.load(..., sr=...))"
            )
        self._samples.append(np.asarray(samples, dtype=np.float32))

    def input_finished(self):
        samples = np.concatenate(self._samples) if self._samples else np.zeros(0)
        # one AcceptWaveform call with the whole utterance, like vosk
        self.online.accept_waveform(self.sample_rate, samples)
        self.online.input_finished()

    @property
    def num_frames_ready(self):
        return self.online.num_frames_ready


# ---------------------------------------------------------------------------
# recognizers
# ---------------------------------------------------------------------------


class OnlineRecognizer:
    """Streaming ASR that decodes like the vosk runtime (see the file
    comment); sherpa-onnx OnlineRecognizer style API.

    The chunk is the model's: kaldi's frames_per_chunk (conf/model.conf
    --frames-per-chunk, default 24) rounded up by GetChunkSize.  One
    decode_stream() call processes one queued 0.2 s step of audio.
    """

    def __init__(self, am):
        self.am = am
        self.model_dir = Path(am.model_dir)
        self.config = VoskConfig(self.model_dir, am.modulus)
        self.mfcc_opts = mfcc_options(self.model_dir / "conf" / "mfcc.conf")
        meta = am.meta
        self.sample_rate = meta["sample_rate"]
        self.factor = meta["frame_subsampling_factor"]
        if self.factor != self.config.frame_subsampling_factor:
            raise ValueError(
                f"frame_subsampling_factor: model {self.factor} != "
                f"conf/model.conf {self.config.frame_subsampling_factor}"
            )
        self.left_context = meta["left_context"]
        self.right_context = meta["right_context"]
        self.chunk_size = self.config.frames_per_chunk
        self.acoustic_scale = self.config.acoustic_scale
        # the model is pure and its output rows are anchored at multiples
        # of `factor`: windows start on that grid
        self._grid_left = round_up(self.left_context, self.factor)

    def create_stream(self):
        return OnlineStream(self)

    def is_ready(self, stream):
        """True while a queued step (or the end-of-input processing) is
        left to do."""
        return bool(stream.pending) or (stream.finished and not stream.finalized)

    def decode_stream(self, stream):
        """Process one queued step (vosk's AcceptWaveform loop body), or,
        once input is finished and the queue is empty, vosk's FinalResult
        (flush features, update silence weights, decode the last chunks,
        finalize)."""
        if stream.pending:
            samples = stream.pending.pop(0)
            stream.mfcc.accept_waveform(
                self.sample_rate, to_int16(samples).astype(np.float32).tolist()
            )
            self._feed_ivector(stream)
            self._update_silence_weights(stream)
            self._advance_decoding(stream)
            return
        if stream.finished and not stream.finalized:
            stream.mfcc.input_finished()
            self._feed_ivector(stream)
            stream.ivector.input_finished()
            stream.flushed = True
            self._update_silence_weights(stream)
            self._advance_decoding(stream)
            stream.decodable.input_is_finished()
            stream.decoder.finalize_decoding()
            stream.finalized = True

    def get_result(self, stream):
        """Partial: the current best path (vosk's PartialResult).  Final
        (after the stream is finalized): the best path of the lattice with
        the graph cost scaled by 0.9 (vosk's GetResult)."""
        decoder = stream.decoder
        if decoder.num_frames_decoded() == 0:
            return ""
        if stream.finalized:
            ok, lat = decoder.get_raw_lattice(use_final_probs=True)
            lm_scale = VOSK_LM_SCALE
        else:
            ok, lat = decoder.get_best_path(use_final_probs=False)
            lm_scale = 1.0
        if not ok:
            return ""
        _, _, word_ids, _ = kd.lattice_best_path(lat, lm_scale=lm_scale)
        return " ".join(stream.words.get(w, f"<{w}>") for w in word_ids if w != 0)

    def reset(self, stream):
        """Start a new utterance on this stream."""
        stream.reset()

    # -- the pipeline steps ---------------------------------------------

    def _feed_ivector(self, stream):
        """New MFCC frames -> the i-vector feature (kaldi's
        OnlineIvectorFeature reads the same base feature)."""
        n = stream.mfcc.num_frames_ready
        if n > stream._iv_fed:
            frames = np.stack(
                [
                    np.array(stream.mfcc.get_frame(i), dtype=np.float32)
                    for i in range(stream._iv_fed, n)
                ]
            )
            stream.ivector.accept_features(frames)
            stream._iv_fed = n

    def _update_silence_weights(self, stream):
        """vosk's Recognizer::UpdateSilenceWeights."""
        n = stream.ivector.num_frames_ready  # the pipeline's NumFramesReady
        if stream.silence.active() and n > 0:
            stream.silence.compute_current_traceback(
                stream.decoder.get_best_path_transition_ids(use_final_probs=False)
            )
            stream.ivector.update_frame_weights(stream.silence.get_delta_weights(n))

    def _decodable_rows_ready(self, stream):
        """kaldi's DecodableNnetLoopedOnlineBase::NumFramesReady: model
        rows the decoder may read."""
        n = stream.mfcc.num_frames_ready
        if n == 0:
            return 0
        if stream.flushed:
            return ceil_div(n, self.factor)
        chunks = max(0, n - self.right_context) // self.chunk_size
        return chunks * self.chunk_size // self.factor

    def _advance_decoding(self, stream):
        """Compute the chunks the decodable needs, feed their rows to the
        decoder (kaldi's AdvanceDecoding: the decoder pulls rows, the
        decodable computes chunks on demand)."""
        target = self._decodable_rows_ready(stream)
        rows_per_chunk = self.chunk_size // self.factor
        new_rows = []
        while stream.num_chunks * rows_per_chunk < target:
            new_rows.append(self._compute_chunk(stream))
        if new_rows:
            # the decodable exposes only `target` rows: the last chunk of
            # the utterance may extend past the end (padded input)
            rows = np.concatenate(new_rows)[: target - stream.num_rows_decoded]
            rows = np.ascontiguousarray(rows, dtype=np.float32)
            if self.acoustic_scale != 1.0:
                rows = rows * self.acoustic_scale
            stream.decodable.accept_loglikes(rows)
            stream.num_rows_decoded += len(rows)
            stream.decoder.advance_decoding(stream.decodable)

    def _compute_chunk(self, stream):
        """kaldi's DecodableNnetLoopedOnlineBase::AdvanceChunk for chunk
        c = stream.num_chunks: take the chunk's i-vector, run the pure
        model on the chunk's window and return the chunk's rows."""
        c = stream.num_chunks
        n_ready = stream.mfcc.num_frames_ready
        # the i-vector from the last input frame we needed
        # ("ivector_frame_to_use"), or zeros if none is ready yet
        iv_ready = stream.ivector.num_frames_ready
        if iv_ready > 0:
            ivector = stream.ivector.get_frame(min(n_ready - 1, iv_ready - 1))
        else:
            ivector = np.zeros(stream.ivector.dim, dtype=np.float32)
        stream.chunker.ivectors.append(np.asarray(ivector, dtype=np.float32))

        # window: the left context on the row grid, the chunk, the right
        # context; frames outside the stream are edge-replicated (kaldi's
        # clamp to [0, num_feature_frames_ready - 1])
        start = c * self.chunk_size - self._grid_left
        n = self._grid_left + self.chunk_size + self.right_context
        last = n_ready - 1
        feats = np.stack(
            [
                np.array(stream.mfcc.get_frame(min(max(t, 0), last)), np.float32)
                for t in range(start, start + n)
            ]
        )
        ivectors = stream.chunker.rows(start, n)
        rows, stream.am_state = self.am.run(
            feats, ivectors, stream.am_state, self.chunk_size
        )
        first = self._grid_left // self.factor
        stream.num_chunks += 1
        return rows[first : first + self.chunk_size // self.factor]


class OfflineRecognizer(OnlineRecognizer):
    """Whole-utterance ASR, sherpa-onnx OfflineRecognizer style:

        rec = OfflineRecognizer(OnnxAm(model_dir))
        stream = rec.create_stream()
        stream.accept_waveform(sample_rate, samples)
        stream.input_finished()
        rec.decode_stream(stream)
        print(rec.get_result(stream))

    vosk decodes a file as one AcceptWaveform(all samples) + FinalResult,
    i.e. the streaming pipeline in 0.2 s steps; this does exactly that, so
    the result equals vosk's (and OnlineRecognizer's)."""

    def create_stream(self):
        return OfflineStream(self)

    def decode_stream(self, stream):
        online = stream.online
        while super().is_ready(online):
            super().decode_stream(online)

    def get_result(self, stream):
        return super().get_result(stream.online)
