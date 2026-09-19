"""Check official preprocessing equivalence and experiment bundle contracts without GPU."""
import ast
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np
from scipy.io import wavfile
from run_sonics_check import windows, normalized, summarize

ROOT=Path(__file__).resolve().parents[1]


class SonicsTests(unittest.TestCase):
    def test_window_coverage_and_official_padding(self):
        for samples in (64000,80000,128000,960000):
            audio=np.linspace(-.2,.3,samples,dtype=np.float32)
            covered=np.zeros(samples,dtype=bool)
            for start,clip in windows(audio,80000):
                self.assertEqual(len(clip),80000)
                covered[start:min(start+80000,samples)]=True
                np.testing.assert_array_equal(clip[:min(80000,samples-start)],audio[start:start+80000])
            self.assertTrue(covered.all())
        tree=ast.parse((ROOT/'data/research/sonics-20260919/upstream/sonics/utils/dataset.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AudioDataset')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='crop_or_pad')
        namespace={'np':np}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'official_crop','exec'),namespace)
        audio=np.linspace(-.1,.1,960000,dtype=np.float32)
        official=namespace['crop_or_pad'](None,audio,1920000,False)
        np.testing.assert_array_equal(windows(audio,1920000)[0][1],official)
        np.testing.assert_array_equal(normalized(official),official/np.maximum(np.std(official),1e-6))
        self.assertTrue(np.isfinite(normalized(np.zeros(80000,dtype=np.float32))).all())

    def test_metric_separation(self):
        rows=[dict(model='alpha-5s',view=view,cohort=cohort,file_fake=label,mean_prob=float(label),max_prob=float(label),reference_baseline=.5)
            for view in ('original','mp3_64k') for cohort in ('external','development_music','development_controls') for label in (0,1)]
        metrics=summarize(rows)
        self.assertEqual(len(metrics),6)
        for name,group in metrics.items():
            self.assertEqual(group['mean_prob']['file_eer'],0)
            self.assertEqual(group['mean_prob']['file_auc'],1)
            self.assertEqual('reference_df_arena' in group,'/original/' in name)

    def test_bundle(self):
        bundle=ROOT/'sonics_check_bundle.zip'
        nb=json.loads((ROOT/'notebooks/colab_sonics_check.ipynb').read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertIn(hashlib.sha256(bundle.read_bytes()).hexdigest(),code)
        self.assertIn('timm==1.0.15',code)
        for c in nb['cells']:
            if c['cell_type']=='code': ast.parse(''.join(c['source']))
        with zipfile.ZipFile(bundle) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
            plan=json.loads(z.read('plan.json'))
            self.assertFalse(plan['fit'])
            self.assertEqual(len(plan['rows']),128)
            for relative,sha in plan['file_sha256'].items():
                self.assertEqual(hashlib.sha256(z.read(relative)).hexdigest(),sha)
                if relative.endswith('.py'): ast.parse(z.read(relative))
            counts={}
            for row in plan['rows']:
                counts[row['cohort']]=counts.get(row['cohort'],0)+1
                blob=z.read(row['path'])
                self.assertEqual(hashlib.sha256(blob).hexdigest(),row['clip_sha256'])
                sr,audio=wavfile.read(io.BytesIO(blob))
                self.assertEqual(audio.ndim,1)
                self.assertTrue(np.isfinite(audio).all())
                self.assertIn(sr,(16000,22050))
            self.assertEqual(counts,dict(development_music=48,development_controls=32,external=48))
            self.assertEqual(plan['expected_predictions'],2*len(plan['rows'])+2*counts['external'])
            for name in ('run_sonics_check.py','run_file_probe_experiment.py'):
                self.assertEqual(z.read(name),(ROOT/'scripts'/name).read_bytes())
            for name in z.namelist():
                if name.startswith('sonics/'):
                    self.assertEqual(z.read(name),(ROOT/'data/research/sonics-20260919/upstream'/name).read_bytes())


if __name__=='__main__':
    unittest.main()
