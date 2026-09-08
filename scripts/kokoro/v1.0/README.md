# Introduction

This directory is for kokoro v1.0

`generate_voices_bin.py` generates the Sherpa-ONNX voice table from the
official `Kokoro-82M/voices` directory. The generated v1.0 table contains 54
voices, including the Spanish `em_santa` voice at speaker ID 53. Existing
speaker IDs 0 through 52 remain unchanged.

The generator validates the style embedding shape (`510 x 1 x 256`) and
`float32` dtype before writing the binary file. Run it from this directory
after downloading the Kokoro-82M repository:

```bash
python3 generate_voices_bin.py
```

The corresponding model metadata must be generated with `add_meta_data.py`,
which derives `n_speakers`, `speaker_names`, `id2speaker`, and `speaker2id`
from the same mapping.
