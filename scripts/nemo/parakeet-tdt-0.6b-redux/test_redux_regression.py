import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file, save_file


path = Path(__file__).with_name('export_onnx.py')
spec = importlib.util.spec_from_file_location('redux_export', path)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class FakeModel:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return self.state


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReduxTests(unittest.TestCase):
    def test_decode_against_scalar_spec_and_padding(self):
        # Two rows and both padding patterns, with a group boundary at col 128.
        for n in (129, 131):
            rows = [[(r * 2 + c) % 3 for c in range(n)] for r in range(2)]
            packed = torch.zeros((2, (n + 4) // 5), dtype=torch.uint8)
            for r in range(2):
                for c in range(n):
                    packed[r, c // 5] += rows[r][c] * 3 ** (c % 5)
            scales = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            expected = torch.tensor([
                [(code - 1) * scales[r, c // 128].item()
                 for c, code in enumerate(row)]
                for r, row in enumerate(rows)
            ])
            actual = export.decode_ternary(packed, scales, n, 128)
            self.assertTrue(torch.equal(actual, expected))
            for value in (1, 2):
                altered = packed.clone()
                altered[:, -1] += value * sum(3 ** i for i in range(n % 5, 5))
                self.assertTrue(torch.equal(
                    export.decode_ternary(altered, scales, n, 128), expected))

    def test_mapping_and_rejections_preserve_files(self):
        name = 'encoder.layers.0.conv.pointwise_conv1'
        weight, bias = name + '.weight', name + '.bias'
        q, scales = name + '.qweight', name + '.scales'
        manifest = {
            'format': 'thrush-ternary-v2', 'quant': {'group_size': 128},
            'quantized_modules': [dict(name=name, in_features=6,
                                       out_features=2, as_conv1d=True)],
        }
        weights = {q: torch.zeros(2, 2, dtype=torch.uint8),
                   scales: torch.ones(2, 1), bias: torch.ones(2)}
        expected = {weight: torch.zeros(2, 6, 1), bias: torch.zeros(2)}
        result = export.redux_state_for(FakeModel(expected), weights, manifest)
        self.assertEqual(tuple(result[weight].shape), (2, 6, 1))
        self.assertTrue(torch.equal(result[bias], weights[bias]))

        cases = [
            ('conv target', weights, {**expected, weight: torch.zeros(2, 6, 2)},
             f'{weight} -> {weight}: 2D (2, 6) is not the trailing-unit 3D conv weight v3 has (2, 6, 2)'),
            ('dense shape', {**weights, bias: torch.ones(3)}, expected,
             f'{bias} -> {bias}: shape (3,), v3 has (2,)'),
            ('both packed and dense', {**weights, weight: torch.ones(2, 6)}, expected,
             f'{weight} is both packed and dense in the file'),
            ('stray packed', {**weights, 'stray.qweight': torch.zeros(1)}, expected,
             "packed tensors that ternary.json does not list: ['stray.qweight']"),
            ('missing packed', {scales: weights[scales], bias: weights[bias]}, expected,
             f"ternary.json names tensors that are not in the file: ['{q}']"),
            ('missing target', {q: weights[q], scales: weights[scales]}, expected,
             f"unexpected: [] (0); missing: ['{bias}'] (1)"),
            ('bad row bytes', {**weights, q: torch.zeros(2, 1, dtype=torch.uint8)}, expected,
             'row_bytes 1 != ceil(6/5)'),
            ('bad scales', {**weights, scales: torch.ones(2, 2)}, expected,
             'scales (2, 2) != (2, 1) for 2x6 in groups of 128'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for label, given, model, message in cases:
                with self.subTest(label=label):
                    data = tmp / 'input.safetensors'
                    targets = tmp / 'expected.safetensors'
                    rules = tmp / 'ternary.json'
                    save_file(given, data)
                    save_file(model, targets)
                    rules.write_text(json.dumps(manifest))
                    before = [digest(p) for p in (data, targets, rules)]
                    with self.assertRaises(SystemExit) as exc:
                        export.redux_state_for(FakeModel(load_file(targets)),
                                               load_file(data),
                                               export.read_ternary_spec(rules))
                    self.assertEqual(str(exc.exception), message)
                    self.assertEqual([digest(p) for p in (data, targets, rules)], before)
                    print(label, ':', str(exc.exception), '; sha256 unchanged')


class ArtifactAndWorkflowTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('REDUX_ENCODER_INT8_ONNX'),
                         'set REDUX_ENCODER_INT8_ONNX to a generated encoder')
    def test_generated_encoder_metadata(self):
        import onnx
        model = onnx.load(os.environ['REDUX_ENCODER_INT8_ONNX'],
                          load_external_data=False)
        metadata = {p.key: p.value for p in model.metadata_props}
        self.assertIn('tdt', metadata['url'].lower())
        self.assertEqual(metadata['model_type'], 'EncDecRNNTBPEModel')
        self.assertLessEqual(model.ir_version, onnx.IR_VERSION)
        onnx_opsets = [p.version for p in model.opset_import if not p.domain]
        self.assertTrue(onnx_opsets)
        self.assertTrue(all(0 < v <= onnx.defs.onnx_opset_version()
                            for v in onnx_opsets))

    def test_redux_package_wiring(self):
        import yaml
        root = path.resolve().parents[3]
        workflow = root / '.github/workflows/export-nemo-parakeet-tdt-0.6b-v2.yaml'
        job = yaml.safe_load(workflow.read_text())['jobs']['export-nemo-parakeet-tdt-0_6b']
        self.assertIn('redux', job['strategy']['matrix']['version'])
        step = next(s for s in job['steps'] if s.get('if') == "matrix.version == 'redux'")
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            script = stage / 'scripts/nemo/parakeet-tdt-0.6b-redux/run.sh'
            script.parent.mkdir(parents=True)
            (script.parent / 'README.md').write_text('model card\n')
            script.write_text('#!/bin/bash\nset -e\nfor f in encoder.onnx encoder.weights '
                              'decoder.onnx joiner.onnx encoder.int8.onnx '
                              'decoder.int8.onnx joiner.int8.onnx tokens.txt en.wav; '
                              'do echo fixture > "$f"; done\n')
            script.chmod(0o755)
            subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', step['run']],
                           cwd=stage, check=True, stdout=subprocess.DEVNULL)
            for label in ('Collect files (fp32)', 'Collect files (int8)'):
                command = next(s['run'] for s in job['steps'] if s.get('name') == label)
                command = command.replace('${{ matrix.version }}', 'redux')
                subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', command],
                               cwd=stage, check=True, stdout=subprocess.DEVNULL)
            prefix = 'sherpa-onnx-nemo-parakeet-tdt-0.6b-redux'
            self.assertTrue((stage / prefix / 'encoder.weights').is_file())
            self.assertTrue((stage / (prefix + '-int8') / 'encoder.int8.onnx').is_file())
            self.assertTrue((stage / (prefix + '-int8.tar.bz2')).is_file())
        publisher = next(s for s in job['steps'] if s.get('name') == 'Publish to huggingface')
        command = publisher['with']['command']
        self.assertIn('sherpa-onnx-nemo-parakeet-tdt-0.6b-$version\n', command)
        self.assertIn('sherpa-onnx-nemo-parakeet-tdt-0.6b-$version-int8\n', command)
        release = next(s for s in job['steps'] if s.get('name') == 'Release')
        self.assertEqual(release['with']['file'], './*.tar.bz2')
        self.assertTrue(release['with']['file_glob'])


if __name__ == '__main__':
    unittest.main()
