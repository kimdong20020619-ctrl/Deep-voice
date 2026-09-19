"""Audit a completed generator comparison without replaying GPU inference."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scipy.io import wavfile
from run_generator_check import crop, summarize

ROOT=Path(__file__).resolve().parents[1]


def main():
    candidates=sorted((Path.home()/'Downloads').glob('generator_results*.zip'),key=lambda p:p.stat().st_mtime,reverse=True)
    source=candidates[0]
    with zipfile.ZipFile(source) as z,zipfile.ZipFile(ROOT/'generator_check_bundle.zip') as bundle:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        plan=json.loads(z.read('plan.json'))
        assert plan==json.loads(bundle.read('plan.json'))
        for name in ('script.py','run_generator_check.py','run_sonics_check.py','run_file_probe_experiment.py'):
            assert z.read(name)==bundle.read(name)
        for name in bundle.namelist():
            if name.startswith('provenance/'): assert z.read(name)==bundle.read(name)
        assert json.loads(z.read('weight_checks.json'))==plan['weight_sha256']
        runtime=json.loads(z.read('runtime.json'))
        assert runtime['fit'] is False
        indexed={r['ID']:r for r in plan['rows']}
        inputs=json.loads(z.read('inputs.json'))
        expected={identity+'_'+str(seconds)+'s_'+view for identity in indexed for seconds in (5,8) for view in ('original','mp3_64k')}
        assert len(inputs)==len(expected)==336 and {r['key'] for r in inputs}==expected
        tasks={r['key']:r for r in inputs}
        for row in indexed.values():
            blob=bundle.read(row['path'])
            assert hashlib.sha256(blob).hexdigest()==row['clip_sha256']
            sr,audio=wavfile.read(io.BytesIO(blob))
            assert sr==16000 and audio.dtype==np.int16
            for seconds in (5,8):
                values,start=crop(audio.astype(np.float32)/32768.,seconds)
                task=tasks[row['ID']+'_'+str(seconds)+'s_original']
                assert task['crop_start_samples']==start
                assert task['decoded_sha256']==hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()
        df=json.loads(z.read('df_predictions.json'))
        assert set(df)==expected
        rows=list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
        assert len(rows)==336
        seen=set()
        max_error=0.
        for row in rows:
            for name in ('file_fake','seconds_input'): row[name]=int(row[name])
            for name in ('sonics_mean','df_arena','sonics_seconds','df_seconds'):
                row[name]=float(row[name]); assert np.isfinite(row[name])
            key=row['ID']+'_'+str(row['seconds_input'])+'s_'+row['view']
            assert key not in seen
            seen.add(key)
            task=tasks[key]
            for name,value in indexed[row['ID']].items(): assert task[name]==value
            for name in ('ID','generator','historical_split','file_fake','seconds_input','view'): assert row[name]==task[name]
            assert row['df_arena']==df[key]['df_arena'] and row['df_seconds']==df[key]['df_seconds']
            assert row['df_seconds']>=0 and row['sonics_seconds']>=0
            assert all(0<=row[k]<=1 for k in ('df_arena','sonics_mean'))
            with np.load(io.BytesIO(z.read('segments/'+key+'.npz')),allow_pickle=False) as detail:
                assert str(detail['decoded_sha256'])==task['decoded_sha256']
                np.testing.assert_array_equal(detail['starts'],[0] if row['seconds_input']==5 else [0,48000])
                logits=detail['logits']
                assert logits.shape==detail['starts'].shape and np.isfinite(logits).all()
                score=float((1/(1+np.exp(-np.clip(logits.astype(float),-700,700)))).mean())
                max_error=max(max_error,abs(score-row['sonics_mean']))
                np.testing.assert_allclose(score,row['sonics_mean'],rtol=1e-12,atol=1e-12)
        assert seen==expected
        assert {n for n in z.namelist() if n.startswith('segments/') and n.endswith('.npz')}=={'segments/'+k+'.npz' for k in expected}
        metrics=summarize(rows)
        assert metrics==json.loads(z.read('summary.json'))['metrics']
        log=z.read('run.log').decode()
        assert 'SONICS 336 / 336' in log and 'DF 336 / 336' in log and 'Traceback (most recent call last)' not in log
        audit=dict(source_filename=source.name,result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            verified_conditions=336,verified_original_waveforms=168,max_sonics_reconstruction_error=max_error,
            metrics=metrics,runtime=runtime,
            inference_seconds={k:sum(r[k] for r in rows) for k in ('sonics_seconds','df_seconds')},
            independently_recomputed_gpu=False,independently_recomputed_mp3=False,
            weight_bytes_returned=False,weight_verification='Expected hashes match recorded checks; identical worker checks before/after inference.',
            decision='No universal replacement; generator-dependent tradeoffs and codec sensitivity remain.')
    out=ROOT/'data/experiments/generator-20260919'
    out.mkdir(parents=True,exist_ok=True)
    target=out/'generator_results.zip'
    if target.exists(): assert target.read_bytes()==source.read_bytes()
    else: shutil.copy2(source,target)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='metrics'},indent=2))


if __name__=='__main__':
    main()
