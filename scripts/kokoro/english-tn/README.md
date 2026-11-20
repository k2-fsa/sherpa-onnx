# English number and symbol rules

This pack expands written English numbers and symbols into words through the
existing `rule_fsts` setting. It composes NVIDIA NeMo English taggers with their
verbalizers, so the result contains spoken words rather than tagged records.
No changes to the runtime, C API, or language bindings are required.

The two FSTs run in order:

```
--tts-rule-fsts=./english-tn/english_tn.fst,./english-tn/english_symbols.fst
```

`english_tn.fst` covers cardinals, decimals, currency, percentages, times,
ordinals and contextual years. `english_symbols.fst` handles standalone AM/PM
and common symbols such as `+`, `=`, `&`, `@` and `/`. For example:

| Input | Output |
|---|---|
| The budget is 1969 dollars. | The budget is one thousand nine hundred and sixty nine dollars. |
| The price is $12.50 and the increase is 15%. | The price is twelve dollars fifty cents and the increase is fifteen percent. |
| In 1969, people watched the moon landing. | In nineteen sixty nine, people watched the moon landing. |
| We meet at 9:30 AM. | We meet at nine thirty a m. |
| We meet at 12:00 PM. | We meet at twelve p m. |
| 3:00 pm | three p m |
| 9:00 | nine o'clock |
| I am available at 9:30 am. | I am available at nine thirty a m. |
| AM-PM | a m-p m |
| US$10 | ten us dollars |
| café3.14 | café3.14 |
| The bin 1969 is full. | The bin one thousand nine hundred and sixty nine is full. |
| 2 + 2 = 4. | two plus two equals four. |

Times use NeMo's deterministic tagger/verbalizer before general number
rewriting. Whole hours omit the zero minutes: a suffix gives `three p m`,
while a bare time gives `nine o'clock`. NeMo accepts lowercase `am`/`pm` in
a time and canonicalizes them for the symbol pass. Bare prose such as
`I am calm` is unchanged. Complete time fields are protected from competing
cardinal or nested-minute rewrites.

The bare `$` currency uses `dollar(s)`; explicit `US$` retains NeMo's
`us dollar(s)` reading. Contextual years require a separate `in` or `year`
word. Boundaries match complete punctuation, symbol or space characters,
preserving number suffixes attached to UTF-8 letters or combining marks
(`café3.14`) while expanding a number after a separator (`café—3.14`).

This is a measured subset of NeMo. Dates and fractions currently produce
component-number readings with spoken slashes; URL/version handling is not
complete. `test_cases.json` distinguishes exact expectations from observed
structured-form behavior. Kokoro's existing OOV policy is unchanged: the
lexicon must contain the spoken words, or callers can supply phonemes/tokens.

## Reproduce

Use Python 3.12 and an isolated virtual environment. On Linux with a compatible
Pynini wheel:

```
python3.12 -m venv .venv
```

On macOS, Pynini builds from source and needs the OpenFst headers/libraries
and Xcode command-line tools. Pynini 2.1.7 targets OpenFst 1.8.4
([version compatibility](https://github.com/kylebgorman/pynini/blob/master/NEWS)).
For example, with micromamba installed:

```
toolchain_dir="$PWD/openfst-toolchain"
micromamba create -y -p "$toolchain_dir" -c conda-forge python=3.12 openfst=1.8.4
"$toolchain_dir/bin/python" -m venv .venv
export CPLUS_INCLUDE_PATH="$toolchain_dir/include"
export LIBRARY_PATH="$toolchain_dir/lib"
export LDFLAGS="-L$toolchain_dir/lib -Wl,-rpath,$toolchain_dir/lib"
```

Then, from this directory:

```
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
git clone https://github.com/NVIDIA/NeMo-text-processing.git
git -C NeMo-text-processing checkout ddadfb2a38d2bc6b8cc6232c4f915eb60f500688
```

The requirements pin NeMo's import dependencies (`regex`, `tqdm`, `joblib`,
`sacremoses`, the eager audio imports `editdistance`/`cdifflib`, and the
`click`/`cloudpickle` dependencies) as well as Pynini/kaldifst. NeMo is
imported from this unmodified source checkout; a full NeMo installation is
unnecessary. `--nemo-source` names the directory containing
`nemo_text_processing/`. The revision also appears in `manifest.json`.

```
python build.py --nemo-source ./NeMo-text-processing --output ./english-tn
python build_symbols.py ./english-tn/english_symbols.fst
python test_runtime.py --bundle ./english-tn --output ./normalization-results.json
python test_clock.py --nemo-source ./NeMo-text-processing \
  --bundle ./english-tn --output ./clock-results.json
python test_package.py
python package.py --bundle ./english-tn --output ./english-tn.tar.bz2
```

The generated rules use standard tropical OpenFst byte labels and load through
sherpa's existing FST loader. Python, Pynini and NeMo are build dependencies;
applications need only the generated FSTs.

## Distribution and Kokoro example

Sherpa examples fetch models, lexicons and normalization FSTs from GitHub
release assets (`tts-models`, `asr-models` or `hr-files`) rather than storing
those payloads in source control. `package.py` follows that convention and
creates a deterministic `english-tn.tar.bz2` containing both compiled rules,
their hashes and notices. It writes a unique temporary file beside the output
and atomically replaces the output after success, preserving a previous
archive if writing fails. The accompanying local PR artifact contains this
compiled pack. Publishing a release asset is a separate maintainer action;
there is no unapproved or placeholder download URL.

After building or unpacking the rules, run the Kokoro example:

```
python python-api-examples/test-offline-tts-kokoro-en-rule-fsts.py \
  --model-dir ./kokoro-en-v0_19 --lexicon ./lexicon-en-us.txt \
  --rules-dir ./english-tn --text 'The budget is 1969 dollars.'
```

NeMo's Apache-2.0 license is retained in `LICENSE-NeMo`. Keep it and `NOTICE`
with generated rule packs. Source and artifact pins are in `manifest.json`.
