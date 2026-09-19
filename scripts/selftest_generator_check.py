"""Check matched durations, per-generator populations and packaging contracts."""
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np
from scipy.io import wavfile
from run_generator_check import crop,summarize

ROOT=Path(__file__).resolve().parents[1]


class GeneratorTests(unittest.TestCase):
    def test_same_source_crop(self):
        audio=np.arange(128000,dtype=np.float32)
        short,start=crop(audio,5)
        full,begin=crop(audio,8)
        self.assertEqual((start,begin),(24000,0))
        np.testing.assert_array_equal(short,full[24000:104000])
        self.assertFalse(np.shares_memory(full,audio))
        with self.assertRaises(AssertionError): crop(audio[:64000],5)

    def test_metric_population(self):
        rows=[]
        for seconds in (5,8):
            for view in ('original','mp3_64k'):
                for label,generator in ((0,'human'),(0,'human'),(1,'A'),(1,'B')):
                    rows.append(dict(seconds_input=seconds,view=view,file_fake=label,generator=generator,sonics_mean=float(label),df_arena=.5))
        metrics=summarize(rows)
        self.assertEqual(len(metrics),8)
        for group in metrics.values():
            self.assertEqual(group['sonics_mean']['n'],3)
            self.assertEqual(group['sonics_mean']['file_eer'],0)
            self.assertEqual(group['df_arena']['file_eer'],.5)

    def test_bundle(self):
        path=ROOT/'generator_check_bundle.zip'
        with zipfile.ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
            plan=json.loads(z.read('plan.json'))
            self.assertEqual(len(plan['rows']),84)
            self.assertEqual(len({r['source_sha256'] for r in plan['rows']}),84)
            self.assertEqual(Counter(r['generator'] for r in plan['rows']),dict(human=41,musicldm=9,MusicGen_medium=7,mustango=11,audioldm2=8,stable_audio_open=8))
            self.assertEqual(plan['expected_predictions'],84*2*2)
            for name,sha in plan['file_sha256'].items():
                blob=z.read(name)
                self.assertEqual(hashlib.sha256(blob).hexdigest(),sha)
                if name.endswith('.py'): ast.parse(blob)
            for row in plan['rows']:
                blob=z.read(row['path'])
                self.assertEqual(hashlib.sha256(blob).hexdigest(),row['clip_sha256'])
                sr,a=wavfile.read(io.BytesIO(blob))
                self.assertEqual((sr,a.shape,a.dtype),(16000,(128000,),np.dtype('int16')))
                for seconds in (5,8):
                    values,_=crop(a.astype(np.float32)/32768,seconds)
                    self.assertTrue(np.isfinite(values).all())
                    self.assertEqual(len(values),16000*seconds)
            self.assertEqual(z.read('run_generator_check.py'),(ROOT/'scripts/run_generator_check.py').read_bytes())
        nb=json.loads((ROOT/'notebooks/colab_generator_check.ipynb').read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(),code)
        self.assertIn('/content/generator_results',code)
        self.assertIn('run_generator_check.py',code)
        for c in nb['cells']:
            if c['cell_type']=='code': ast.parse(''.join(c['source']))


if __name__=='__main__':
    unittest.main()
