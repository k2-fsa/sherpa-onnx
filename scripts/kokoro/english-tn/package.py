#!/usr/bin/env python3
# Copyright (c) 2026 Hedy AI
"""Package generated rules as a release asset; keep binary payloads out of git."""
import argparse
import hashlib
import io
import json
import pathlib
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
a = p.parse_args()
source = pathlib.Path(__file__).parent
manifest = json.loads((source / 'manifest.json').read_text())
files = {}
for name, expected in manifest['artifacts'].items():
    data = (a.bundle / name).read_bytes()
    assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
    files[name] = data
files['LICENSE-NeMo'] = (source / 'LICENSE-NeMo').read_bytes()
files['NOTICE'] = (source / 'NOTICE').read_bytes()
files['manifest.json'] = (source / 'manifest.json').read_bytes()
a.output.parent.mkdir(parents=True, exist_ok=True)
with tarfile.open(a.output, 'w:bz2', format=tarfile.USTAR_FORMAT) as archive:
    for name, data in sorted(files.items()):
        info = tarfile.TarInfo('english-tn/' + name)
        info.size = len(data)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(data))
print(hashlib.sha256(a.output.read_bytes()).hexdigest(), a.output)
