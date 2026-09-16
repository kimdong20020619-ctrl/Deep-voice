"""Test split/label guards and real sklearn training with mocked GPU embeddings."""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from scipy.io import wavfile
from run_mixed_probe import run, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def example():
    sources, rows = [], []
    for split in ('train', 'development'):
        for family in ('speech','music','mix_voice_-6dB','mix_voice_+6dB'):
            for label in (0,1):
                identity = f'{split}_{family}_{label}'
                source = dict(ID=identity, split=split, kind='music', file_fake=label,
                              source_group=identity, source_sha256=identity, generator=split)
                sources.append(source)
                rows.append(dict(ID=identity, split=split, family=family, file_fake=label,
                    source_ids=[identity], source_groups=[identity], source_hashes=[identity], clip_sha256=identity))
    return dict(sources=sources, rows=rows)


class MixedTests(unittest.TestCase):
    def test_guards(self):
        valid = example()
        validate_manifest(valid)
        for field in ('source_group', 'source_sha256', 'generator'):
            broken = copy.deepcopy(valid)
            broken['sources'][9][field] = broken['sources'][1][field]
            with self.assertRaises(ValueError):
                validate_manifest(broken)
        wrong = copy.deepcopy(valid)
        wrong['rows'][0]['file_fake'] = 1
        with self.assertRaises(ValueError):
            validate_manifest(wrong)
        wrong = copy.deepcopy(valid)
        wrong['rows'][0]['source_ids'] = [wrong['sources'][8]['ID']]
        with self.assertRaises(ValueError):
            validate_manifest(wrong)

    def test_training_orchestration(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work/'probe/test').mkdir(parents=True)
            (work/'model').mkdir()
            (work/'model/SHA256SUMS.txt').write_text('mock')
            shutil.copytree(ROOT/'src', work/'src', ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('run_mixed_probe.py','run_file_probe_experiment.py'):
                shutil.copy2(ROOT/'scripts'/name,work/name)
            manifest = example()
            for i, row in enumerate(manifest['rows']):
                path = work/'probe/test'/(row['ID']+'.wav')
                value = (.1 if row['file_fake'] else -.1) + i*.0001
                wavfile.write(path,16000,np.full(128000,value,dtype=np.float32))
                row['clip_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (work/'probe/manifest.json').write_text(json.dumps(manifest))
            module = ast.parse((ROOT/'submit/script.py').read_text(encoding='utf-8'))
            klass = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='LinearProbe')
            fake = '''import numpy as np
CONFIG = {'file_head':'direct', 'segment_agg':'max'}
class DFArenaScorer:
    has_embeddings = True
    def __init__(self, device): pass
    def score(self, audio, kind=None, want_embeddings=False):
        value = float(audio.mean())
        return float(value>0), np.array([[value,value*2],[value,value*2]], dtype=np.float32)
'''
            (work/'script.py').write_text(fake+'\n'+ast.unparse(klass),encoding='utf-8')
            np.savez(work/'previous_music_probe.npz',w=np.ones(2),b=0.,mean=np.zeros(2),scale=np.ones(2))
            torch = SimpleNamespace(__version__='mock',Tensor=type('Tensor',(),{}),device=lambda x:x,
                cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda n:'mock',empty_cache=lambda:None))
            with patch.dict(sys.modules,torch=torch),contextlib.redirect_stdout(io.StringIO()):
                run(work)
                summary = json.loads((work/'mixed_probe_results/summary.json').read_text())
                self.assertEqual(summary['mixed_probe']['macro_family_eer'],0)
                with self.assertRaises(ValueError):
                    run(work)

    def test_final_bundle(self):
        path = ROOT/'mixed_probe_bundle.zip'
        if not path.exists():
            self.skipTest('Build bundle after official source collection')
        nb = json.loads((ROOT/'notebooks/colab_mixed_probe.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
        self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(),code)
        self.assertIn('/content/mixed_probe_results',code)
        self.assertNotIn('file_mixed_probe_results',code)
        with zipfile.ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            manifest = json.loads(z.read('probe/manifest.json'))
            validate_manifest(manifest)
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
            for row in manifest['rows']:
                blob = z.read('probe/test/'+row['ID']+'.wav')
                self.assertEqual(hashlib.sha256(blob).hexdigest(),row['clip_sha256'])
                sr,audio = wavfile.read(io.BytesIO(blob))
                self.assertEqual((sr,audio.shape,audio.dtype),(16000,(128000,),np.dtype('float32')))
                self.assertTrue(np.isfinite(audio).all())
                self.assertLessEqual(np.abs(audio).max(),.950001)
            self.assertEqual(z.read('run_mixed_probe.py'),(ROOT/'scripts/run_mixed_probe.py').read_bytes())


if __name__ == '__main__':
    unittest.main()
