"""Validate fixed-weight diagnostic inputs and fail-closed pipeline comparison."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.io import wavfile
from run_frozen_probe_check import run

ROOT = Path(__file__).resolve().parents[1]


class FrozenTests(unittest.TestCase):
    def test_bundle_and_conditions(self):
        nb = json.loads((ROOT / 'notebooks/colab_frozen_probe.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        with zipfile.ZipFile(ROOT / 'frozen_probe_bundle.zip') as z:
            self.assertIsNone(z.testzip())
            m = json.loads(z.read('probe/manifest.json'))
            self.assertEqual(len(m['rows']), 64)
            self.assertEqual({r['split'] for r in m['rows']}, {'development'})
            self.assertEqual(hashlib.sha256(z.read('frozen_probe.npz')).hexdigest(), m['probe_sha256'])
            waves = {}
            for row in m['rows']:
                blob = z.read('probe/test/'+row['ID']+'.wav')
                self.assertEqual(hashlib.sha256(blob).hexdigest(), row['clip_sha256'])
                sr, audio = wavfile.read(io.BytesIO(blob))
                self.assertEqual((sr, audio.shape), (16000, (128000,)))
                self.assertTrue(np.isfinite(audio).all())
                waves[row['ID']] = audio
            for source in {r['original_ID'] for r in m['rows']}:
                expected = waves[source+'_reference'].astype(np.float32)/32768
                np.testing.assert_array_equal(waves[source+'_float_reference'], expected)
                np.testing.assert_array_equal(waves[source+'_quiet'], expected*0.1)

    def check_pipeline(self, mismatch):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work / 'probe/test').mkdir(parents=True)
            (work / 'frozen_probe.npz').write_bytes(b'mock frozen weight')
            rows = []
            for label in (0, 1):
                target = work / 'probe/test' / (str(label)+'.wav')
                wavfile.write(target, 16000, np.full(128000, 1000 if label else -1000, dtype=np.int16))
                rows.append(dict(ID=str(label), split='development', condition='reference', file_fake=label,
                                 clip_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
            (work / 'probe/manifest.json').write_text(json.dumps(dict(rows=rows, probe_sha256=hashlib.sha256(b'mock frozen weight').hexdigest())))
            fake = '''import numpy as np
from scipy.io import wavfile
CONFIG = {'file_head': 'direct'}
def load_audio_16k(path): return wavfile.read(path)[1].astype(float)/32768
def load_panns_model(device): return None
def load_htdemucs_model(device): return None
def resolve_agg(): return 'max'
def aggregate_segment_scores(scores, mode): return max(scores)
class DFArenaScorer:
    def __init__(self, device): pass
    def score(self, audio, want_embeddings=False): return 0.5, np.array([[audio.mean()]])
class LinearProbe:
    def __init__(self, path): pass
    def predict(self, features): return np.where(features[:,0]>0, 0.8, 0.2)
def process_one_file(path, panns, scorer, demucs, device, file_probe=None):
    audio = load_audio_16k(path)
    score = 0.5 if file_probe is None else float(file_probe.predict(np.array([[audio.mean()]]))[0])
    return {'FILE_FAKE_PROB': score, 'VOICE_FAKE_PROB': 0.3}
'''
            if mismatch:
                fake = fake.replace("'FILE_FAKE_PROB': score", "'FILE_FAKE_PROB': score*0.5")
            (work / 'script.py').write_text(fake)
            torch = SimpleNamespace(Tensor=type('Tensor', (), {}), device=lambda x:x,
                                    cuda=SimpleNamespace(is_available=lambda:True))
            with patch.dict(sys.modules, torch=torch), contextlib.redirect_stdout(io.StringIO()):
                if mismatch:
                    with self.assertRaises(AssertionError):
                        run(work)
                    self.assertFalse((work / 'frozen_probe_results/summary.json').exists())
                else:
                    run(work)
                    report = json.loads((work / 'frozen_probe_results/summary.json').read_text())
                    self.assertEqual(report['reference']['probe']['file_eer'], 0)
            self.assertEqual((work / 'frozen_probe.npz').read_bytes(), b'mock frozen weight')

    def test_matching_pipeline(self):
        self.check_pipeline(False)

    def test_mismatching_pipeline_stops(self):
        self.check_pipeline(True)


if __name__ == '__main__':
    unittest.main()
