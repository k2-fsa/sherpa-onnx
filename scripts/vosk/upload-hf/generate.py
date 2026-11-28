#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# The generate.sh driver shared by the two vosk upload flows; run it
# from the flow's own directory (it renders ./generate.sh from
# ./generate.sh.in):
#
#   * upload-hf/   -- mirror the original vosk zips to
#     https://huggingface.co/csukuangfj/vosk-models
#   * upload-onnx/ -- (generate.py is a symlink to this file) export the
#     models to ONNX and upload to
#     https://huggingface.co/csukuangfj/vosk-models-onnx
#
# The model list itself lives in ../vosk_models.py (single source for
# both flows).  Pass --total/--index to split the list across CI runners
# (sharding follows scripts/piper/generate.py).

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import jinja2

# scripts/vosk on sys.path for vosk_models; __file__ resolves through
# the upload-onnx/ symlink to this file, so both flows find it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vosk_models import (  # noqa: E402
    LANG_NAMES,
    TEST_WAV_BASE,
    get_all_models,
    lang_of,
    supports_of,
)

HF_REPO = "csukuangfj/vosk-models"


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
    parser.add_argument(
        "--info",
        action="store_true",
        help="Write release/asr/<lang>/<name>.md info pages for every model "
        "with a local ls -lhR listing (from --listing-dir); generate.sh "
        "uploads each to asr/<lang>/<name>.md on csukuangfj/vosk-models",
    )
    parser.add_argument(
        "--only",
        default="",
        help="With --info: render the info page of just this model "
        "(generate.sh does this per model, right before uploading it)",
    )
    parser.add_argument(
        "--readme",
        action="store_true",
        help="Assemble README.md (tables linking to the asr/<lang>/<name>.md "
        "info pages), "
        "instead of rendering generate.sh",
    )
    parser.add_argument(
        "--listing-dir",
        default="listing",
        help="Directory holding <model>.txt ls -lhR listings",
    )
    parser.add_argument(
        "--upload",
        nargs="+",
        default=[],
        metavar="LOCAL:REMOTE",
        help="Upload the given files to --repo in ONE commit "
        "(the repo allows only 128 commits per hour, so per-file commits "
        "do not scale to every model's zip + info page), then exit",
    )
    parser.add_argument(
        "--repo",
        default=HF_REPO,
        help="Hugging face repo for --upload",
    )
    parser.add_argument(
        "--message",
        default="upload",
        help="Commit message for --upload",
    )
    parser.add_argument(
        "--upload-release",
        nargs="+",
        default=[],
        metavar="LOCAL:NAME",
        help="Upload the given files to the GitHub release --tag of "
        "--gh-repo (same-name assets are overwritten), then exit",
    )
    parser.add_argument(
        "--gh-repo",
        default="k2-fsa/sherpa-onnx",
        help="GitHub repo for --upload-release",
    )
    parser.add_argument(
        "--tag",
        default="asr-models",
        help="GitHub release tag for --upload-release",
    )
    parser.add_argument(
        "--langs",
        default="",
        help="Comma separated language codes to keep, e.g. cn,en,de "
        "(default: every language)",
    )
    parser.add_argument(
        "--prepare-dir",
        default="",
        metavar="NAME",
        help="Add test_wavs/ (the test wav + its transcript) and "
        "README.md (origin + license) to the exported model dir "
        "--dir of model NAME, then exit",
    )
    parser.add_argument(
        "--dir",
        default="",
        help="Model dir for --prepare-dir",
    )
    parser.add_argument(
        "--wav",
        default="",
        help="Test wav for --prepare-dir (default test-wavs/<wav>)",
    )
    return parser.parse_args()


def hf_commit(pairs, message, repo=HF_REPO):
    """Upload all <local>:<remote> pairs to the HF repo in ONE commit."""
    import time

    try:
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:  # huggingface_hub < 0.23
        from huggingface_hub.utils import HfHubHTTPError
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = []
    for pair in pairs:
        local, remote = pair.split(":", 1)
        operations.append(
            CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
        )
    api = HfApi()
    # HF allows only 128 commit creations per hour per repo; when that is
    # used up it answers 429 "retry in about 1 hour" -- so wait it out
    # (minutes-scale bash retries cannot bridge an hour-long window)
    for attempt in range(12):
        try:
            commit_info = api.create_commit(
                repo_id=repo,
                operations=operations,
                commit_message=message,
                token=os.environ["HF_TOKEN"],
            )
            print(f"{commit_info}: {len(operations)} files in one commit")
            return
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if (status != 429 and "429" not in str(e)) or attempt == 11:
                raise
            print(
                f"429 from huggingface (commit rate limit); "
                f"waiting 300s (attempt {attempt + 1}/12)",
                flush=True,
            )
            time.sleep(300)


def gh_release_upload(pairs, gh_repo, tag):
    """Upload all <local>:<name> pairs to the GitHub release `tag` of
    gh_repo (a same-name asset is overwritten)."""
    import time

    import requests

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    assert token, "GH_TOKEN must be set to upload to a GitHub release"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    api = f"https://api.github.com/repos/{gh_repo}"
    r = requests.get(f"{api}/releases/tags/{tag}", headers=headers)
    r.raise_for_status()
    release = r.json()
    upload_url = release["upload_url"].split("{")[0]

    def all_assets():
        """name -> asset id, PAGINATED: the API returns one page
        (30 by default) and a big release carries hundreds of assets --
        an unpaged listing misses most names, the delete is skipped
        and the upload fails 422 'already_exists'."""
        existing = {}
        page = 1
        while True:
            r = requests.get(
                f"{api}/releases/{release['id']}/assets",
                headers=headers,
                params={"per_page": 100, "page": page},
            )
            r.raise_for_status()
            batch = r.json()
            if not batch:
                return existing
            for a in batch:
                existing[a["name"]] = a["id"]
            page += 1

    for pair in pairs:
        local, name = pair.split(":", 1)
        # GitHub rejects release assets >= 2 GB (422 only after the
        # whole file has streamed); such zips are huggingface-only
        size = os.path.getsize(local)
        if size >= 2 * 1024 * 1024 * 1024:
            raise SystemExit(
                f"{local}: {size / 1024 / 1024 / 1024:.2f} GB >= 2 GB -- "
                f"GitHub release assets must be smaller "
                f"(upload to huggingface only)"
            )
        for attempt in range(3):
            existing = all_assets()
            if name in existing:
                requests.delete(
                    f"{api}/releases/assets/{existing[name]}", headers=headers
                ).raise_for_status()
                time.sleep(1)  # deletion is eventually consistent
            with open(local, "rb") as f:
                r = requests.post(
                    upload_url,
                    params={"name": name},
                    headers={**headers, "Content-Type": "application/octet-stream"},
                    data=f,
                )
            if r.status_code != 422:
                r.raise_for_status()
                break
            # 422: the name is still taken (the delete is not visible
            # yet) -- rescan (paginated) and retry
            time.sleep(3)
        else:
            r.raise_for_status()
        print(f"{local} -> github.com/{gh_repo}/releases/{tag}:{name}")


def find_model(name):
    for m in get_all_models():
        if m.name == name:
            return m
    raise SystemExit(f"unknown model: {name}")


def prepare_dir(name, out_dir, wav=""):
    """Add test_wavs/ (the test wav + its transcript) and README.md
    (where the model is from + its license) to an exported model dir."""
    m = find_model(name)
    assert m.wav and m.expect, f"{name}: no test wav for language {m.lang}"
    out_dir = Path(out_dir)
    wav = Path(wav) if wav else Path("test-wavs") / m.wav

    test_wavs = out_dir / "test_wavs"
    test_wavs.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wav, test_wavs / m.wav)
    (test_wavs / "README.md").write_text(
        f"""# Test wav

`{m.wav}` is from [{m.wav}]({TEST_WAV_BASE}/{m.wav}).

Transcript:

{m.expect}
"""
    )

    readme = out_dir / "README.md"  # vosk ships a plain `README`
    if not readme.is_file():
        license_ = m.license or "(not listed on the vosk models page)"
        recurrent_note = ""
        meta_file = out_dir / "am" / "model_meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
        input_dim = meta.get("input_dim", "?")
        ivector_dim = meta.get("ivector_dim", "?")
        num_pdfs = meta.get("num_pdfs", "?")
        factor = int(meta.get("frame_subsampling_factor", 3))
        if int(meta.get("recurrent", 0)):
            rows, dim = meta["state_rows"], meta["state_dim"]
            recurrent_note = f"""- The acoustic model is RECURRENT (LSTM): besides `features` and
  `ivector` it takes `state` [B, {rows}, {dim}] and returns `state_out`.
  Feed zeros for the first window of an utterance (or a one-shot run),
  then each window's `state_out` to the next window.  Windows are
  `round_up(left_context, 3)` + n new + `right_context` frames; see
  `scripts/vosk/nnet3/notes.md` ("Recurrent nets").
"""
        readme.write_text(
            f"""# {name} (ONNX)

ONNX export of the vosk model [`{name}`]({m.url}) (see
https://alphacephei.com/vosk/models).

- License: {license_}
- Converted with `scripts/vosk/` (`export_torch.py` + `export_onnx.py`);
  `am/model_meta.json` describes the model dims and receptive field.
{recurrent_note}- `am/final.onnx` is the float32 acoustic model, `am/final.int8.onnx`
  the quantized (int8) variant (QInt8 weights, per-channel; the input
  affine stays float32); each ships in its own zip.
- Inputs/outputs: `features` [B, T, {input_dim}] and `ivector`
  [B, T, {ivector_dim}] -- one i-vector per input frame (kaldi's online
  decoders give each 24-frame chunk its own i-vector; repeat that chunk's
  i-vector over its frames) -- and `log_probs` [B, ceil(T/{factor}), {num_pdfs}],
  one row per {factor} input frames.  `python3
  scripts/vosk/describe_onnx.py --model-dir <this dir>` prints the full
  signature.
- `test_wavs/` has a test wav and its transcript.

The other files (`conf/`, `ivector/`, `graph/`, `README`) are unchanged
from the original vosk model, except `graph/words.txt`, which is
regenerated from the graph's output symbol table when the original
ids disagree with it (then the info page says so).
"""
        )
    print(f"{out_dir}: test_wavs/ + README.md added")


CONTAINS = {
    "am": "acoustic model (`am/`)",
    "conf": "configs (`conf/`)",
    "graph": "decoding graph (`graph/`)",
    "ivector": "i-vector extractor (`ivector/`)",
    "rescore": "rescoring extras (`rescore/`)",
    "rnnlm": "RNNLM (`rnnlm/`)",
    "test_wavs": "test wav + transcript (`test_wavs/`)",
    "README": "model notes (`README`)",
}


def contains_of(listing: str) -> str:
    """What the model folder contains, from its ls -lhR listing (the
    first block lists the folder's top-level entries)."""
    entries = []
    lines = listing.splitlines()
    i = 1 if lines and lines[0].rstrip().endswith(":") else 0
    for line in lines[i:]:
        if line.rstrip().endswith(":"):
            break  # next folder block: the top-level listing is done
        if not line.strip() or line.startswith("total"):
            continue
        entries.append(line.split()[-1])
    parts = [desc for name, desc in CONTAINS.items() if name in entries]
    return "; ".join(parts) if parts else "(see listing)"


def render_info(all_model_list, listing_dir: Path, only: str = ""):
    """release/asr/<lang>/<name>.md for every model with a local listing:
    what the zip contains and its ls -lhR file sizes.  With `only`, just
    that one model (generate.sh renders + uploads each model as it goes)."""
    listing_dir = Path(listing_dir)
    out = Path("release")  # info pages land next to their zip:
    # release/asr/<lang>/<name>.md -> uploaded as asr/<lang>/<name>.md
    env = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    with open("./info.md.in") as f:
        template = env.from_string(f.read())
    made = 0
    for m in all_model_list:
        if only and m.name != only:
            continue
        p = listing_dir / f"{m.name}.txt"
        if not p.is_file():
            continue
        page = out / "asr" / lang_of(m.name) / f"{m.name}.md"
        page.parent.mkdir(parents=True, exist_ok=True)
        tree_p = listing_dir / f"{m.name}.tree.txt"
        zips_p = listing_dir / f"{m.name}.zips.txt"
        zips_text = zips_p.read_text() if zips_p.is_file() else ""
        io_p = listing_dir / f"{m.name}.io.txt"
        words_p = listing_dir / f"{m.name}.words-note.txt"
        page.write_text(
            template.render(
                name=m.name,
                url=m.url,
                supports=supports_of(m.name),
                contains=contains_of(p.read_text()),
                listing=p.read_text().rstrip(),
                tree=tree_p.read_text().rstrip()
                if tree_p.is_file()
                else "(tree listing not available)",
                license=m.license or "(not listed on the vosk models page)",
                zips=zips_text.rstrip() or "(not available)",
                io=io_p.read_text().rstrip()
                if io_p.is_file()
                else "(signature listing not available)",
                words_note=words_p.read_text().rstrip() if words_p.is_file() else "",
                int8="-int8" in zips_text,  # the int8 zip was produced
            )
        )
        made += 1
    print(f"asr/<lang>/<name>.md: {made} info pages written")


def render_readme(all_model_list):
    """README.md: table of contents and one table per language, each row
    linking to the model's asr/<lang>/<name>.md info page (which lists
    the files in its zip)."""
    by_lang = {}
    for m in all_model_list:
        by_lang.setdefault(lang_of(m.name), []).append(m)
    languages = []
    for lang in sorted(by_lang):
        title = "{} (`{}`)".format(LANG_NAMES.get(lang, lang), lang)
        languages.append(
            {
                "title": title,
                "anchor": re.sub(r"[^a-z0-9 -]", "", title.lower()).replace(" ", "-"),
                "models": [
                    {
                        "name": m.name,
                        "url": m.url,
                        "lang": lang_of(m.name),
                        "supports": supports_of(m.name),
                        "license": m.license,
                    }
                    for m in by_lang[lang]
                ],
            }
        )
    environment = jinja2.Environment(
        trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True
    )
    with open("./README.md.in") as f:
        template = environment.from_string(f.read())
    with open("./README.md", "w") as f:
        f.write(template.render(languages=languages))
    print(f"README.md written for {len(all_model_list)} models")


def main():
    args = get_args()
    if args.upload:
        hf_commit(args.upload, args.message, repo=args.repo)
        return
    if args.upload_release:
        gh_release_upload(args.upload_release, args.gh_repo, args.tag)
        return
    if args.prepare_dir:
        prepare_dir(args.prepare_dir, args.dir, wav=args.wav)
        return
    index = args.index
    total = args.total
    assert 0 <= index < total, (index, total)

    all_model_list = get_all_models()
    if args.langs:
        keep = set(args.langs.split(","))
        all_model_list = [m for m in all_model_list if m.lang in keep]

    # Note: --info/--readme must return BEFORE generate.sh is rendered:
    # generate.sh calls `generate.py --info --only <name>` while itself
    # running, and bash reads its script file incrementally -- rewriting
    # it mid-run would corrupt execution.
    if args.info:
        render_info(all_model_list, Path(args.listing_dir), only=args.only)
        return
    if args.readme:
        render_readme(all_model_list)
        return

    num_models = len(all_model_list)
    num_per_runner = num_models // total
    if num_per_runner <= 0:
        raise ValueError(f"num_models: {num_models}, num_runners: {total}")

    start = index * num_per_runner
    end = start + num_per_runner

    remaining = num_models - total * num_per_runner

    print(f"{index}/{total}: {start}-{end}/{num_models}")

    d = dict()
    d["model_list"] = all_model_list[start:end]

    if index < remaining:
        s = total * num_per_runner + index
        d["model_list"].append(all_model_list[s])
        print(f"{s}/{num_models}")

    filename_list = [
        "./generate.sh",
    ]
    for filename in filename_list:
        environment = jinja2.Environment()
        if not Path(f"{filename}.in").is_file():
            print(f"skip {filename}")
            continue

        with open(f"{filename}.in") as f:
            s = f.read()
        template = environment.from_string(s)

        s = template.render(**d)
        with open(filename, "w") as f:
            print(s, file=f)
        os.chmod(filename, 0o755)  # the rendered script is meant to be run

    print(f"There are {len(all_model_list)} models")
    for m in d["model_list"]:
        print(m.index, m.kind, m.name)


if __name__ == "__main__":
    main()
