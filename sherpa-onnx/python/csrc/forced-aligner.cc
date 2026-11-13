// sherpa-onnx/python/csrc/forced-aligner.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/python/csrc/forced-aligner.h"

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/forced-aligner.h"

namespace sherpa_onnx {

static constexpr const char *kAlignmentIntervalDoc = R"doc(
One time interval of an alignment tier.

For example, the word "ask" of a 3.85 s utterance::

  AlignmentInterval(start=0.06, end=0.45, text="ask")
)doc";

static constexpr const char *kAlignmentIntervalStartDoc = R"doc(
Start time of this interval in seconds.
)doc";

static constexpr const char *kAlignmentIntervalEndDoc = R"doc(
End time of this interval in seconds.
)doc";

static constexpr const char *kAlignmentIntervalTextDoc = R"doc(
Label of this interval: a phone name, a word/character, or an empty
string for silence gaps.
)doc";

static constexpr const char *kAlignmentTierDoc = R"doc(
One tier of an alignment: a named sequence of time intervals.
)doc";

static constexpr const char *kAlignmentTierNameDoc = R"doc(
Name of this tier, e.g., ``"phones"`` or ``"words"``.
)doc";

static constexpr const char *kAlignmentTierIntervalsDoc = R"doc(
Intervals of this tier in time order. They tile the whole utterance:
adjacent intervals share an endpoint and silence is represented by
intervals with an empty label.
)doc";

static constexpr const char *kForcedAlignmentResultDoc = R"doc(
Result of :meth:`ForcedAligner.compute`.

It contains the phone and word boundaries of one utterance and can be
serialized to a Praat TextGrid via :meth:`to_textgrid`.
)doc";

static constexpr const char *kForcedAlignmentResultDurationDoc = R"doc(
Duration of the aligned utterance in seconds.
)doc";

static constexpr const char *kForcedAlignmentResultTiersDoc = R"doc(
Alignment tiers. ``tiers[0]`` is the ``"phones"`` tier and ``tiers[1]``
is the ``"words"`` tier. The result is empty (no tiers) when the
alignment failed; see :meth:`ForcedAligner.compute`.

For example::

  for tier in result.tiers:            # "phones", then "words"
      for interval in tier.intervals:  # in time order
          print(tier.name, interval.start, interval.end, interval.text)

  # phones 0.0 0.06 ''              (silence gap)
  # phones 0.06 0.25 æ              (a phone of the word "ask")
  # ...
  # words  0.0 0.06 ''              (silence gap)
  # words  0.06 0.45 ask            (one word interval)
)doc";

static constexpr const char *kForcedAlignmentResultToTextGridDoc = R"doc(
Return the alignment as a Praat TextGrid string (long format).

Returns:
  The TextGrid as a string. Write it to a ``.TextGrid`` file and open it
  with Praat (https://www.fon.hum.uva.nl/praat/) to visualize the
  alignment (or use any other TextGrid reader). It looks like::

    File type = "ooTextFile"
    Object class = "TextGrid"

    xmin = 0
    xmax = 3.8500
    tiers? <exists>
    size = 2
    item []:
        item [1]:
            class = "IntervalTier"
            name = "phones"
            ...
        item [2]:
            class = "IntervalTier"
            name = "words"
            ...
)doc";

static constexpr const char *kForcedAlignmentOptionsDoc = R"doc(
Per-utterance alignment options, following kalpy's GmmAligner defaults.

An instance of this class is the ``options`` field of
:class:`ForcedAlignerConfig`; a stream can override any of the fields
via :meth:`OfflineStream.set_option` before
:meth:`ForcedAligner.compute`, e.g. ``s.set_option("beam", "20")``.
)doc";

static constexpr const char *kForcedAlignmentOptionsBeamDoc = R"doc(
Decoding beam used for the first alignment attempt. Default: 10.
)doc";

static constexpr const char *kForcedAlignmentOptionsRetryBeamDoc = R"doc(
Decoding beam used for the second attempt when the first one fails.
Default: 40.
)doc";

static constexpr const char *kForcedAlignmentOptionsTransitionScaleDoc =
    R"doc(
Scale applied to the transition probabilities of the HMM topology.
Default: 1.
)doc";

static constexpr const char *kForcedAlignmentOptionsSelfLoopScaleDoc =
    R"doc(
Scale applied to the self-loop transition probabilities. Default: 0.1.
)doc";

static constexpr const char *kForcedAlignmentOptionsAcousticScaleDoc =
    R"doc(
Scale applied to the acoustic (GMM) log-likelihoods. Default: 0.1.
)doc";

static constexpr const char *kForcedAlignerConfigDoc = R"doc(
Configuration for :class:`ForcedAligner`.

It points at a forced-alignment acoustic model directory and a
pronunciation dictionary, and carries the default alignment options.
)doc";

static constexpr const char *kForcedAlignerConfigModelDirDoc = R"doc(
Directory of one forced-alignment acoustic model. Required files:

  - ``meta.json``: feature chain + silence/OOV parameters (parsed once
    at construction, field by field).
  - ``final.alimdl``: alignment model, kaldi format (TransitionModel +
    AmDiagGmm). Note: NOT ``final.mdl``.
  - ``tree``: context dependency tree (ContextDependency).
  - ``lda.mat``: LDA transform of the feature chain; required iff the
    feature chain ends in splice+LDA.

Optional files:

  - ``phones.txt``: phone table (symbol to id). When missing, the phone
    ids are taken from the ``phones`` list of ``meta.json`` following
    the kaldi convention of these models (``<eps>=0``, ``sil=1``,
    ``spn=2``, then the lexical phones from 3). Models that do not
    follow this convention are rejected with an error.

A model directory looks like::

  english-mfa-v3.1.0/
    meta.json       # feature chain + silence/OOV parameters
    final.alimdl    # alignment model (kaldi TransitionModel + AmDiagGmm)
    tree            # context dependency tree
    lda.mat         # LDA transform of the feature chain
    phones.txt      # phone table (optional)
    lexicon.txt     # pronunciation dictionary (not read from here; see
                    # the lexicon field of this config)
    en.wav          # test wav (added by the model package)
    README.md       # where the model comes from

Unused files of the upstream model bundles (``final.mdl``,
``phone_lm.fst``, ``graphemes.txt``, ``rules.yaml``, ...) are ignored.

Model packages ready to use (each contains a test wav and a lexicon)
are at https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
)doc";

static constexpr const char *kForcedAlignerConfigLexiconDoc = R"doc(
Pronunciation dictionary, one pronunciation per line::

  key [probability columns ...] phone1 phone2 ...

Probability columns (floating point numbers between the key and the
phones, e.g. MFA ``.dict`` files) are ignored. Several lines under one
key = several pronunciations of the same word/character, and ALL are
kept: each pronunciation becomes a parallel path in the lexicon FST and
the acoustics pick the one used by the alignment. This is how Chinese
polyphones avoid text-level guessing.

The key shape depends on the language: English = words, Chinese = hanzi
characters/phrases (hanzi-keyed, e.g. the ``lexicon.txt`` of the model
packages), other languages = their writing units. Multi-word phrases
keep their combined readings; ``" || "`` separates the per-word (or
per-character) phone groups.

Note: word intervals are recovered along the pronunciation path
actually taken by the alignment; they are not counted with the phone
count of the first pronunciation (pronunciations differ in length).
Words without a pronunciation are reported as out-of-vocabulary and
aligned as the model's OOV phone.

A lexicon looks like::

  ask æ s k              # one word, one pronunciation
  can k ə n              # a second word
  can kʰ æ n             # a second pronunciation of the same word
  行 x a ŋ               # Chinese: one line per reading (polyphones)
  行 h a ŋ
  研究 i n ɛ n || tɕ i o u  # per-character phone groups of a phrase
)doc";

static constexpr const char *kForcedAlignerConfigOptionsDoc = R"doc(
Default alignment options. A stream can override any of them via
:meth:`OfflineStream.set_option` before :meth:`ForcedAligner.compute`.
)doc";

static constexpr const char *kForcedAlignerConfigDebugDoc = R"doc(
True to print debug information: model info while loading the model
and per-utterance alignment info in :meth:`ForcedAligner.compute`. It
can also be enabled for a single stream via
``s.set_option("debug", "1")``.
)doc";

static constexpr const char *kForcedAlignerDoc = R"doc(
Forced aligner: given a wave file and its transcript, it computes the
start/end time of every phone and word.

Usage::

  config = sherpa_onnx.ForcedAlignerConfig()
  config.model_dir = "/path/to/english-mfa-v3.1.0"
  config.lexicon = "/path/to/english-mfa-v3.1.0/lexicon.txt"

  aligner = sherpa_onnx.ForcedAligner(config)
  s = aligner.create_stream()
  s.accept_waveform(sample_rate, samples)
  result = aligner.compute(s, text)
  print(result.to_textgrid())
)doc";

static constexpr const char *kForcedAlignerInitDoc = R"doc(
Constructor for ForcedAligner. It loads the acoustic model, the tree
and the pronunciation dictionary.

Args:
  config:
    Config for forced alignment. See :class:`ForcedAlignerConfig`.
)doc";

static constexpr const char *kForcedAlignerCreateStreamDoc = R"doc(
Create an utterance stream.

Accept a waveform with :meth:`OfflineStream.accept_waveform` (float32
samples normalized to the range [-1, 1], i.e., raw int16 samples divided
by 32768), optionally override alignment options with
:meth:`OfflineStream.set_option`, then pass the stream to
:meth:`compute`.

Returns:
  An :class:`OfflineStream` instance.
)doc";

static constexpr const char *kForcedAlignerComputeDoc = R"doc(
Compute the alignment of one utterance.

Args:
  stream:
    A stream created by :meth:`create_stream` and filled with a waveform.
  text:
    The transcript of the whole utterance. Hanzi text is tokenized per
    character (multi-character dictionary keys give pronunciation
    context); anything else is split on whitespace. Punctuation is
    ignored.

Returns:
  The :class:`ForcedAlignmentResult` of this utterance. It is empty (no
  tiers) with a warning on stderr when the stream is null, the text is
  empty, the stream has no features, or word boundaries cannot be
  recovered from the alignment.
)doc";

static void PybindAlignmentInterval(py::module *m) {
  using PyClass = AlignmentInterval;
  py::class_<PyClass>(*m, "AlignmentInterval", kAlignmentIntervalDoc)
      .def_readwrite("start", &PyClass::start, kAlignmentIntervalStartDoc)
      .def_readwrite("end", &PyClass::end, kAlignmentIntervalEndDoc)
      .def_readwrite("text", &PyClass::text, kAlignmentIntervalTextDoc);
}

static void PybindAlignmentTier(py::module *m) {
  PybindAlignmentInterval(m);
  using PyClass = AlignmentTier;
  py::class_<PyClass>(*m, "AlignmentTier", kAlignmentTierDoc)
      .def_readwrite("name", &PyClass::name, kAlignmentTierNameDoc)
      .def_readwrite("intervals", &PyClass::intervals,
                     kAlignmentTierIntervalsDoc);
}

static void PybindForcedAlignmentResult(py::module *m) {
  PybindAlignmentTier(m);
  using PyClass = ForcedAlignmentResult;
  py::class_<PyClass>(*m, "ForcedAlignmentResult", kForcedAlignmentResultDoc)
      .def_readwrite("duration", &PyClass::duration,
                     kForcedAlignmentResultDurationDoc)
      .def_readwrite("tiers", &PyClass::tiers, kForcedAlignmentResultTiersDoc)
      .def("to_textgrid", &PyClass::ToTextGrid,
           kForcedAlignmentResultToTextGridDoc)
      .def("__str__", &PyClass::ToTextGrid);
}

static void PybindForcedAlignmentOptions(py::module *m) {
  using PyClass = ForcedAlignmentOptions;
  py::class_<PyClass>(*m, "ForcedAlignmentOptions", kForcedAlignmentOptionsDoc)
      .def(py::init<>())
      .def_readwrite("beam", &PyClass::beam,
                     kForcedAlignmentOptionsBeamDoc)
      .def_readwrite("retry_beam", &PyClass::retry_beam,
                     kForcedAlignmentOptionsRetryBeamDoc)
      .def_readwrite("transition_scale", &PyClass::transition_scale,
                     kForcedAlignmentOptionsTransitionScaleDoc)
      .def_readwrite("self_loop_scale", &PyClass::self_loop_scale,
                     kForcedAlignmentOptionsSelfLoopScaleDoc)
      .def_readwrite("acoustic_scale", &PyClass::acoustic_scale,
                     kForcedAlignmentOptionsAcousticScaleDoc)
      .def("__str__", &PyClass::ToString);
}

static void PybindForcedAlignerConfig(py::module *m) {
  PybindForcedAlignmentOptions(m);
  using PyClass = ForcedAlignerConfig;
  py::class_<PyClass>(*m, "ForcedAlignerConfig", kForcedAlignerConfigDoc)
      .def(py::init<>())
      .def_readwrite("model_dir", &PyClass::model_dir,
                     kForcedAlignerConfigModelDirDoc)
      .def_readwrite("lexicon", &PyClass::lexicon,
                     kForcedAlignerConfigLexiconDoc)
      .def_readwrite("options", &PyClass::options,
                     kForcedAlignerConfigOptionsDoc)
      .def_readwrite("debug", &PyClass::debug, kForcedAlignerConfigDebugDoc)
      .def("validate", &PyClass::Validate)
      .def("__str__", &PyClass::ToString);
}

void PybindForcedAligner(py::module *m) {
  PybindForcedAlignerConfig(m);
  PybindForcedAlignmentResult(m);

  using PyClass = ForcedAligner;
  py::class_<PyClass>(*m, "ForcedAligner", kForcedAlignerDoc)
      .def(py::init<const ForcedAlignerConfig &>(), py::arg("config"),
           kForcedAlignerInitDoc, py::call_guard<py::gil_scoped_release>())
      .def("create_stream", &PyClass::CreateStream,
           kForcedAlignerCreateStreamDoc,
           py::call_guard<py::gil_scoped_release>())
      .def("compute", &PyClass::Compute, py::arg("stream"), py::arg("text"),
           kForcedAlignerComputeDoc, py::call_guard<py::gil_scoped_release>());
}

}  // namespace sherpa_onnx
