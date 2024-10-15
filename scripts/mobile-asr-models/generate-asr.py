#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import jinja2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total",
        type=int,
        default=1,
        help="Number of runners",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the current runner",
    )
    return parser.parse_args()


@dataclass
class Model:
    # We will download
    # https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_name}.tar.bz2
    model_name: str

    cmd: str


def get_streaming_zipformer_transducer_models():
    models = [
        Model(
            model_name="sherpa-onnx-streaming-zipformer-korean-2024-06-16",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-20-avg-1-chunk-16-left-128.onnx \
              --output1 $dst/encoder-epoch-20-avg-1-chunk-16-left-128.onnx \
              --output2 $dst/encoder-epoch-20-avg-1-chunk-16-left-128.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/README.md $dst/
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-20-avg-1-chunk-16-left-128.onnx $dst/
            cp -v $src/joiner-epoch-20-avg-1-chunk-16-left-128.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
            """,
        ),
        Model(
            model_name="icefall-asr-zipformer-streaming-wenetspeech-20230615",
            cmd="""
            ./run-impl.sh \
              --input $src/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
              --output1 $dst/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
              --output2 $dst/encoder-epoch-12-avg-4-chunk-16-left-128.int8.onnx

            cp -fv $src/README.md $dst/
            cp -v $src/data/lang_char/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx $dst/
            cp -v $src/exp/joiner-epoch-12-avg-4-chunk-16-left-128.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-en-2023-06-26",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
              --output1 $dst/encoder-epoch-99-avg-1-chunk-16-left-128.onnx \
              --output2 $dst/encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/README.md $dst/
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-99-avg-1-chunk-16-left-128.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
              """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-en-2023-06-21",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

            cp -fv $src/README.md $dst/
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
                """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-en-2023-02-21",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/README.md $dst/ || true
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
              """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

            cp -v $src/README.md $dst/
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-fr-2023-04-14",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-29-avg-9-with-averaged-model.onnx \
              --output1 $dst/encoder-epoch-29-avg-9-with-averaged-model.onnx \
              --output2 $dst/encoder-epoch-29-avg-9-with-averaged-model.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/README.md $dst/ || true
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-29-avg-9-with-averaged-model.onnx $dst/
            cp -v $src/joiner-epoch-29-avg-9-with-averaged-model.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
              """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

            mkdir $dst/{64,96}

            ./run-impl.sh \
              --input $src/64/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/64/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/64/encoder-epoch-99-avg-1.int8.onnx

            ./run-impl.sh \
              --input $src/96/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/96/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/96/encoder-epoch-99-avg-1.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/README.md $dst/ || true
            cp -av $src/test_wavs $dst/

            cp -v $src/tokens.txt $dst/
            cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

            cp -v $src/tokens.txt $dst/64/
            cp -v $src/64/decoder-epoch-99-avg-1.onnx $dst/64/
            cp -v $src/64/joiner-epoch-99-avg-1.int8.onnx $dst/64/

            cp -v $src/tokens.txt $dst/96/
            cp -v $src/96/decoder-epoch-99-avg-1.onnx $dst/96/
            cp -v $src/96/joiner-epoch-99-avg-1.int8.onnx $dst/96/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
              """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/README.md $dst/ || true
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
            """,
        ),
        Model(
            model_name="sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-99-avg-1.onnx \
              --output1 $dst/encoder-epoch-99-avg-1.onnx \
              --output2 $dst/encoder-epoch-99-avg-1.int8.onnx

            cp -v $src/bpe.model $dst/ || true
            cp -v $src/README.md $dst/ || true
            cp -v $src/tokens.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-99-avg-1.onnx $dst/
            cp -v $src/joiner-epoch-99-avg-1.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
            """,
        ),
    ]

    return models


def get_models():
    return get_streaming_zipformer_transducer_models()


def main():
    args = get_args()
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_models()

    num_models = len(all_model_list)

    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - args.total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")

    d = dict()
    d["model_list"] = all_model_list[start:end]
    if index < remaining:
        s = args.total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    filename_list = [
        "./run2.sh",
    ]
    for filename in filename_list:
        environment = jinja2.Environment()
        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)


if __name__ == "__main__":
    main()
