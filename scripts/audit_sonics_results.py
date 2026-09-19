"""Reconstruct frozen SONICS scores from returned segment logits."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from run_sonics_check import summarize

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=Path.home()/'Downloads/sonics_results.zip'
    with zipfile.ZipFile(source) as z, zipfile.ZipFile(ROOT/'sonics_check_bundle.zip') as bundle:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        plan=json.loads(z.read('plan.json'))
        assert plan==json.loads(bundle.read('plan.json'))
        for name in ('run_sonics_check.py','run_file_probe_experiment.py'):
            assert z.read(name)==bundle.read(name)
        for name in bundle.namelist():
            if name.startswith('provenance/'):
                assert z.read(name)==bundle.read(name),name
        indexed={r['ID']:r for r in plan['rows']}
        transforms=json.loads(z.read('transforms.json'))
        assert len(transforms)==128 and {r['ID'] for r in transforms}==set(indexed)
        transforms={r['ID']:r for r in transforms}
        runtime=json.loads(z.read('runtime.json'))
        assert runtime['fit'] is False and runtime['timm']=='1.0.15'
        expected={(r['ID'],model,view) for r in plan['rows'] for model in plan['models']
            for view in ('original','mp3_64k') if model=='alpha-5s' or r['cohort']=='external'}
        rows=list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
        assert len(rows)==len(expected)==plan['expected_predictions']==352
        assert {(r['ID'],r['model'],r['view']) for r in rows}==expected
        assert {n for n in z.namelist() if n.startswith('segments/') and n.endswith('.npz')}=={
            'segments/'+identity+'_'+model+'_'+view+'.npz' for identity,model,view in expected}
        max_error=0.
        for row in rows:
            original=indexed[row['ID']]
            row['file_fake']=int(row['file_fake'])
            row['segments']=int(row['segments'])
            for key in ('mean_prob','max_prob','reference_baseline','seconds'):
                row[key]=float(row[key])
                assert np.isfinite(row[key])
            assert row['seconds']>=0
            for key in ('cohort','family','file_fake','reference_baseline'):
                assert row[key]==original[key],key
            t=transforms[row['ID']]
            assert t['original_samples']==(960000 if row['cohort']=='external' else 128000)
            assert abs(t['mp3_samples']-t['original_samples'])<=1600
            for key in ('mp3_sha256','decoded_sha256'):
                assert len(t[key])==64 and all(c in '0123456789abcdef' for c in t[key])
            name='segments/'+row['ID']+'_'+row['model']+'_'+row['view']+'.npz'
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as detail:
                logits=detail['logits']
                samples=int(detail['input_samples'])
                assert samples==t['original_samples' if row['view']=='original' else 'mp3_samples']
                assert logits.ndim==1 and len(logits)==row['segments'] and np.isfinite(logits).all()
                length=80000 if row['model']=='alpha-5s' else 1920000
                starts=list(range(0,max(1,samples-length+1),length))
                if samples>length and starts[-1]!=samples-length:
                    starts.append(samples-length)
                np.testing.assert_array_equal(detail['starts'],starts)
                assert len(starts)==len(logits)
                probabilities=1/(1+np.exp(-np.clip(logits.astype(float),-700,700)))
                for key,value in [('mean_prob',float(probabilities.mean())),('max_prob',float(probabilities.max()))]:
                    max_error=max(max_error,abs(value-row[key]))
                    np.testing.assert_allclose(value,row[key],rtol=1e-12,atol=1e-12)
                    assert 0<=row[key]<=1
        metrics=summarize(rows)
        assert metrics==json.loads(z.read('summary.json'))['metrics']
        log=z.read('run.log').decode()
        assert 'done 352 / 352' in log and 'Traceback (most recent call last)' not in log
        audit=dict(result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),verified_inputs=128,
            verified_predictions=352,maximum_score_reconstruction_error=max_error,metrics=metrics,runtime=runtime,
            seconds_by_model={m:sum(r['seconds'] for r in rows if r['model']==m) for m in plan['models']},
            model_weight_bytes_returned=False,weight_check_basis='Matching worker asserts expected SHA before and after inference; completion log and summary present.',
            gpu_logits_independently_recomputed=False,mp3_audio_independently_recomputed=False,
            decision='Primary candidate fails prespecified cross-corpus improvement gate; retain as research candidate, do not replace submission.')
    out=ROOT/'data/experiments/sonics-20260919'
    out.mkdir(parents=True,exist_ok=True)
    target=out/source.name
    if target.exists():
        assert target.read_bytes()==source.read_bytes()
    else:
        shutil.copy2(source,target)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
