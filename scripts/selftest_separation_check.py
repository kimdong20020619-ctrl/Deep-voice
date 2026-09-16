"""Check populations, level controls, and diagnostic execution without a real GPU."""
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
from run_separation_check import level_control, validate_plan, run

ROOT = Path(__file__).resolve().parents[1]


class SeparationTests(unittest.TestCase):
    def test_level_control(self):
        audio = np.sin(np.arange(128000)*.01).astype(np.float32)*.2
        output,gain,silent = level_control(audio,1e-5)
        self.assertFalse(silent)
        self.assertAlmostEqual(float(np.sqrt(np.mean(output.astype(float)**2))),.05,places=7)
        np.testing.assert_allclose(output,audio*gain)
        zeros = np.zeros(128000,dtype=np.float32)
        output,gain,silent = level_control(zeros,1e-5)
        self.assertTrue(silent)
        self.assertEqual(gain,1)
        np.testing.assert_array_equal(output,zeros)
        impulse = zeros.copy()
        impulse[0]=1
        self.assertLessEqual(float(np.abs(level_control(impulse,1e-5)[0]).max()),.950001)
        with self.assertRaises(ValueError):
            level_control(np.array([np.nan]),1e-5)

    def test_bundle_and_population(self):
        path = ROOT/'separation_bundle.zip'
        notebook = json.loads((ROOT/'notebooks/colab_separation_check.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(),code)
        self.assertIn('/content/separation_results',code)
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(cell['source']))
        with zipfile.ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
            plan = json.loads(z.read('probe/manifest.json'))
            validate_plan(plan)
            self.assertEqual(len(plan['rows']),80)
            self.assertEqual(sum(r['music_comparison_eligible'] for r in plan['rows']),48)
            self.assertEqual(hashlib.sha256(z.read('fixed_music_probe.npz')).hexdigest(),plan['probe_sha256'])
            self.assertEqual(z.read('run_separation_check.py'),(ROOT/'scripts/run_separation_check.py').read_bytes())
            self.assertNotIn('experimental_mixed_probe.npz',z.namelist())
            for row in plan['rows']:
                self.assertEqual(hashlib.sha256(z.read('probe/test/'+row['ID']+'.wav')).hexdigest(),row['clip_sha256'])
            invalid = copy.deepcopy(plan)
            invalid['rows'][0]['music_comparison_eligible'] = not invalid['rows'][0]['music_comparison_eligible']
            with self.assertRaises(ValueError):
                validate_plan(invalid)
            invalid = copy.deepcopy(plan)
            invalid['rows'][0]['split'] = 'train'
            with self.assertRaises(ValueError):
                validate_plan(invalid)

    def exercise_pipeline(self, missing_embeddings=False):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            (work/'probe/test').mkdir(parents=True)
            (work/'model').mkdir()
            (work/'model/SHA256SUMS.txt').write_text('mock')
            for name in ('run_separation_check.py','run_file_probe_experiment.py'):
                shutil.copy2(ROOT/'scripts'/name,work/name)
            np.savez(work/'fixed_music_probe.npz',w=np.ones(2),b=0.,mean=np.zeros(2),scale=np.ones(2))
            rows,sources = [],[]
            for kind in ('music','speech'):
                for label in (0,1):
                    identity = kind+str(label)
                    path = work/'probe/test'/(identity+'.wav')
                    wavfile.write(path,16000,np.full(128000,.1 if label else -.1,dtype=np.float32))
                    sources.append(dict(ID=identity,kind=kind,file_fake=label,split='development'))
                    rows.append(dict(ID=identity,family=kind,file_fake=label,split='development',source_ids=[identity],
                                     music_comparison_eligible=kind=='music',clip_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            plan = dict(rows=rows,sources=sources,probe_sha256=hashlib.sha256((work/'fixed_music_probe.npz').read_bytes()).hexdigest(),
                        reference_predictions={r['ID']:dict(baseline=r['file_fake'],previous_music_probe=.5) for r in rows})
            (work/'probe/manifest.json').write_text(json.dumps(plan))
            module = ast.parse((ROOT/'submit/script.py').read_text(encoding='utf-8'))
            klass = next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='LinearProbe')
            fake = '''import numpy as np
from scipy.io import wavfile
CONFIG = {'demucs_gating':True,'gate_voice':.2,'gate_music':.1}
def resolve_agg(kind): return 'max'
def load_audio_16k(path): return wavfile.read(path)[1]
def calculate_rms(audio): return float(np.sqrt(np.mean(audio.astype(float)**2)))
def silence_threshold(): return 1e-5
def load_htdemucs_model(device): return None
def load_panns_model(device): return None
def predict_presence(panns,audio): return .9,.9
def separate_voice_and_music(audio,model,device): return audio*.75,audio*.25
class DFArenaScorer:
    has_embeddings=True
    def __init__(self,device): pass
    def score(self,audio,kind=None,want_embeddings=False):
        value=float(audio.mean())
        return float(value>0), np.array([[value,value*2]],dtype=np.float32)
'''
            if missing_embeddings:
                fake=fake.replace('np.array([[value,value*2]],dtype=np.float32)','None')
            (work/'script.py').write_text(fake+'\n'+ast.unparse(klass),encoding='utf-8')
            torch = SimpleNamespace(__version__='mock',Tensor=type('Tensor',(),{}),device=lambda x:x,
                cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda n:'mock'))
            with patch.dict(sys.modules,torch=torch),contextlib.redirect_stdout(io.StringIO()):
                if missing_embeddings:
                    with self.assertRaises(AssertionError):
                        run(work)
                    self.assertFalse((work/'separation_results/summary.json').exists())
                else:
                    run(work)
                    report=json.loads((work/'separation_results/summary.json').read_text())
                    self.assertEqual(report['eligible_file_metrics']['music']['music_raw']['probe']['file_eer'],0)
                    self.assertNotIn('speech',report['eligible_file_metrics'])
                    self.assertEqual(report['excluded_controls']['speech']['n'],2)
                    self.assertEqual(len(list((work/'separation_results/features').glob('*.npz'))),12)
                    with self.assertRaises(ValueError):
                        run(work)

    def test_execution(self):
        self.exercise_pipeline()

    def test_non_silent_embedding_failure_stops(self):
        self.exercise_pipeline(True)


if __name__=='__main__':
    unittest.main()
