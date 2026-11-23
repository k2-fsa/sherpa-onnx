#!/usr/bin/env python3
"""Exercise the file replay CLI with local tiny ONNX fixtures."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import wave


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    args = p.parse_args()
    model = args.fixtures.resolve(strict=True)
    command = [str(args.binary.resolve(strict=True)),
        f'--qwen3-asr-conv-frontend={model}/conv.onnx',
        f'--qwen3-asr-encoder={model}/encoder.onnx',
        f'--qwen3-asr-decoder={model}/eos.onnx',
        f'--qwen3-asr-tokenizer={model}/tokenizer', '--num-threads=1']
    with tempfile.TemporaryDirectory(prefix='qwen-cli-') as tmp:
        wav = Path(tmp) / 'audio.wav'
        for n, ends in [(1,[1]), (31999,[31999]), (64000,[32000,64000,64000]),
                         (100800,[32000,64000,96000,100800])]:
            with wave.open(str(wav),'wb') as w:
                w.setparams((1,2,16000,n,'NONE','not compressed'))
                w.writeframes(struct.pack('<h',1000)*n)
            for mode in ['none','rollback']:
                run = subprocess.run(command+[f'--prefix-mode={mode}',str(wav)],
                    capture_output=True,check=True,timeout=60)
                rows=[json.loads(s) for s in run.stdout.splitlines()]
                assert [r['audio_end_sample'] for r in rows]==ends, rows
                assert sum(r['is_final'] for r in rows)==1
                assert rows[-1]['is_final']
                assert all(r['status']=='ok' for r in rows)
                assert all('result' in r and 'had_invalid_utf8' in r for r in rows)
                assert [r['revision'] for r in rows]==list(range(1,len(rows)+1))
        for option, status in [('--qwen3-asr-max-total-len=10','context_limit'),
                               ('--max-utterance-samples=64000','audio_limit')]:
            run=subprocess.run(command+[option,str(wav)],capture_output=True,
                check=False,timeout=60)
            assert run.returncode!=0
            rows=[json.loads(s) for s in run.stdout.splitlines()]
            assert rows[-1]['status']==status,rows
            assert not any(r['is_final'] for r in rows)
        # Odd-sized RIFF metadata is padded. Decode from the validated data
        # offset, rather than passing it to a second parser with other rules.
        canonical=wav.read_bytes()
        padded=bytearray(canonical[:36]+b'JUNK'+struct.pack('<I',1)+b'x\0'+canonical[36:])
        padded[4:8]=struct.pack('<I',len(padded)-8)
        wav.write_bytes(padded)
        run=subprocess.run(command+[str(wav)],capture_output=True,check=True,timeout=60)
        rows=[json.loads(s) for s in run.stdout.splitlines()]
        assert rows[-1]['audio_end_sample']==100800 and rows[-1]['is_final']
        # A small malicious header must not allocate according to a forged size.
        data=bytearray(canonical[:44]); data[40:44]=struct.pack('<I',2**31)
        wav.write_bytes(data)
        run=subprocess.run(command+[str(wav)],capture_output=True,check=False,timeout=60)
        assert run.returncode!=0
    print('CLI tail, exact boundary, limits and malformed WAV: passed')


if __name__=='__main__':
    main()
