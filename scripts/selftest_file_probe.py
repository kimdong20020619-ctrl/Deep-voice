"""Check leakage guards and train/evaluate orchestration without claiming GPU validation."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
from run_file_probe_experiment import validate_rows, run
from prepare_probe_dataset import normalized_crop

REPO = Path(__file__).resolve().parents[1]


def example_rows():
    return [dict(ID=f'{split}_{label}', split=split, file_fake=label, group=f'{split}_{label}',
                 source_sha256=f's_{split}_{label}', clip_sha256=f'c_{split}_{label}', generator=split)
            for split in ('train', 'development', 'holdout') for label in (0, 1)]


class ProbeTests(unittest.TestCase):
    def test_same_normalization_and_nonfinite_rejection(self):
        wave = np.sin(np.arange(128000)*0.01).astype(np.float32)
        results = []
        for scale in (0.25, 2.0):
            buffer = io.BytesIO()
            wavfile.write(buffer, 16000, wave*scale)
            results.append(normalized_crop(buffer.getvalue())[0])
        np.testing.assert_array_equal(*results)
        self.assertLessEqual(np.abs(results[0].astype(float)).max(), 31130)
        bad = io.BytesIO()
        wavfile.write(bad, 16000, np.full(128000, np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            normalized_crop(bad.getvalue())

    def test_leakage_guards(self):
        valid = example_rows()
        validate_rows(valid)
        for field in ('group', 'source_sha256', 'clip_sha256', 'generator'):
            rows = copy.deepcopy(valid)
            rows[3][field] = rows[1][field]
            with self.assertRaises(ValueError, msg=field):
                validate_rows(rows)
        with self.assertRaises(ValueError):
            validate_rows(valid[:-1])

    def test_pipeline_with_mock_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'model').mkdir()
            (root / 'model/SHA256SUMS.txt').write_text('mock')
            (root / 'probe/test').mkdir(parents=True)
            rows = example_rows()
            for i, row in enumerate(rows):
                path = root / 'probe/test' / (row['ID'] + '.wav')
                pcm = np.full(128000, (1000 if row['file_fake'] else -1000) + i, dtype=np.int16)
                wavfile.write(path, 16000, pcm)
                row['clip_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / 'probe/manifest.json').write_text(json.dumps({'rows': rows}))
            model_ast = ast.parse((REPO / 'submit/script.py').read_text(encoding='utf-8'))
            probe_class = next(n for n in model_ast.body if isinstance(n, ast.ClassDef) and n.name == 'LinearProbe')
            fake = '''import numpy as np
CONFIG = {'file_head': 'direct', 'segment_agg': 'max'}
class DFArenaScorer:
    has_embeddings = True
    def __init__(self, device): pass
    def score(self, audio, kind=None, want_embeddings=False):
        mean = float(audio.mean())
        return float(mean > 0), np.array([[mean, mean*2], [mean, mean*2]])
'''
            (root / 'script.py').write_text(fake + '\n' + ast.unparse(probe_class), encoding='utf-8')
            sys.path.insert(0, str(REPO / 'src'))
            fake_torch = SimpleNamespace(__version__='mock', Tensor=type('Tensor', (), {}), device=lambda x: x,
                cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda n: 'mock GPU', empty_cache=lambda: None))
            with patch.dict(sys.modules, torch=fake_torch), contextlib.redirect_stdout(io.StringIO()):
                run(root)
                report = json.loads((root / 'probe_results/holdout.json').read_text())
                self.assertEqual(report['holdout']['probe']['file_eer'], 0)
                self.assertTrue((root / 'probe_results/selection.json').exists())
                with self.assertRaises(ValueError):
                    run(root)


if __name__ == '__main__':
    unittest.main()
