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


def get_kws_models():
    models = [
        Model(
            model_name="sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
              --output1 $dst/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
              --output2 $dst/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx

            cp -v $src/README.md $dst/
            cp -v $src/*.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-12-avg-2-chunk-16-left-64.onnx $dst/
            cp -v $src/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
                  """,
        ),
        Model(
            model_name="sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
            cmd="""
            ./run-impl.sh \
              --input $src/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
              --output1 $dst/encoder-epoch-12-avg-2-chunk-16-left-64.onnx \
              --output2 $dst/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx

            cp -v $src/bpe.model $dst/
            cp -v $src/README.md $dst/
            cp -v $src/*.txt $dst/
            cp -av $src/test_wavs $dst/
            cp -v $src/decoder-epoch-12-avg-2-chunk-16-left-64.onnx $dst/
            cp -v $src/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx $dst/

            cat > $dst/notes.md <<EOF
# Introduction
This model is converted from
https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/$src.tar.bz2
and it supports only batch size equal to 1.
EOF
                  """,
        ),
    ]
    return models


def get_models():
    return get_kws_models()


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
