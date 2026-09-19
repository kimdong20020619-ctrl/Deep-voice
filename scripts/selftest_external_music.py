"""Validate new-source pairing and frozen inference orchestration with mocked GPU."""
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
from run_external_music_check import run, validate_plan, summarize

ROOT=Path(__file__).resolve().parents[1]


def example():
    return dict(rows=[dict(ID=f'{pair}_{label}',pair_id=pair,file_fake=label,split='external_check',
        music_seconds=10.,music_source_group=pair,music_source_tokens=[pair],speech_source_ids=[pair+'_speech'],
        clip_sha256=f'{pair}_{label}') for pair in ('A','B') for label in (0,1)])


class ExternalTests(unittest.TestCase):
    def test_pair_guards(self):
        plan=example()
        validate_plan(plan)
        for change in ('music','split','pair','hash','overlap'):
            bad=copy.deepcopy(plan)
            if change=='music': bad['rows'][0]['music_seconds']=0
            if change=='split': bad['rows'][0]['split']='train'
            if change=='pair': bad['rows'][0]['file_fake']=1
            if change=='hash': bad['rows'][0]['clip_sha256']=bad['rows'][1]['clip_sha256']
            if change=='overlap':
                for row in bad['rows'][2:]: row['music_source_tokens']=['A']
            with self.assertRaises(ValueError,msg=change): validate_plan(bad)

    def test_frozen_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            work=Path(temporary)
            (work/'probe/test').mkdir(parents=True)
            (work/'model').mkdir()
            (work/'model/SHA256SUMS.txt').write_text('mock')
            for name in ('run_external_music_check.py','run_file_probe_experiment.py'):
                shutil.copy2(ROOT/'scripts'/name,work/name)
            plan=example()
            plan['model_sha256']={}
            for name in ('music_probe','mixed_probe'):
                path=work/(name+'.npz')
                np.savez(path,w=np.ones(2),b=0.,mean=np.zeros(2),scale=np.ones(2))
                plan['model_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
            for i,row in enumerate(plan['rows']):
                path=work/'probe/test'/(row['ID']+'.wav')
                wavfile.write(path,16000,np.full(960000,(.1 if row['file_fake'] else -.1)+i*.0001,dtype=np.float32))
                row['clip_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            (work/'probe/manifest.json').write_text(json.dumps(plan))
            module=ast.parse((ROOT/'submit/script.py').read_text(encoding='utf-8'))
            klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='LinearProbe')
            fake='''import numpy as np
from scipy.io import wavfile
CONFIG={'file_head':'direct'}
def resolve_agg(kind): return 'max'
def load_audio_16k(path): return wavfile.read(path)[1]
class DFArenaScorer:
    has_embeddings=True
    def __init__(self,device): pass
    def score(self,audio,kind=None,want_embeddings=False):
        value=float(audio.mean())
        return float(value>0),np.array([[value,value*2]],dtype=np.float32)
'''
            (work/'script.py').write_text(fake+'\n'+ast.unparse(klass),encoding='utf-8')
            torch=SimpleNamespace(__version__='mock',Tensor=type('Tensor',(),{}),device=lambda x:x,
                cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda n:'mock'))
            with patch.dict(sys.modules,torch=torch),contextlib.redirect_stdout(io.StringIO()):
                run(work)
                summary=json.loads((work/'external_music_results/summary.json').read_text())
                self.assertEqual(summary['file_metrics']['music_probe']['file_eer'],0)
                self.assertEqual(summary['file_metrics']['music_probe']['paired_fake_ranked_higher'],2)
                with self.assertRaises(ValueError): run(work)
            for name,sha in plan['model_sha256'].items():
                self.assertEqual(hashlib.sha256((work/(name+'.npz')).read_bytes()).hexdigest(),sha)

    def test_bundle(self):
        path=ROOT/'external_music_bundle.zip'
        if not path.exists(): self.skipTest('Official collection pending')
        nb=json.loads((ROOT/'notebooks/colab_external_music.ipynb').read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(),code)
        self.assertIn('/content/external_music_results',code)
        for cell in nb['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
        with zipfile.ZipFile(path) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()),len(set(z.namelist())))
            plan=json.loads(z.read('probe/manifest.json'))
            validate_plan(plan)
            self.assertEqual(len(plan['rows']),48)
            for row in plan['rows']:
                blob=z.read('probe/test/'+row['ID']+'.wav')
                self.assertEqual(hashlib.sha256(blob).hexdigest(),row['clip_sha256'])
                sr,audio=wavfile.read(io.BytesIO(blob))
                self.assertEqual((sr,audio.ndim,audio.dtype),(22050,1,np.dtype('int16')))
                self.assertTrue(59*sr<=len(audio)<=61*sr)
                meta=z.read('probe/metadata/'+row['ID']+'.json')
                self.assertEqual(hashlib.sha256(meta).hexdigest(),row['metadata_sha256'])
            for name,sha in plan['model_sha256'].items():
                self.assertEqual(hashlib.sha256(z.read(name+'.npz')).hexdigest(),sha)
            self.assertEqual(z.read('run_external_music_check.py'),(ROOT/'scripts/run_external_music_check.py').read_bytes())


if __name__=='__main__':
    unittest.main()
