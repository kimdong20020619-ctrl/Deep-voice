"""Recheck returned separated waveforms, frozen probe predictions and populations."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from scipy.io import wavfile
from run_separation_check import VIEWS, level_control, validate_plan, summarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from train.fit_probe import predict


def main():
    source = Path.home()/'Downloads/separation_results.zip'
    with zipfile.ZipFile(source) as z, zipfile.ZipFile(ROOT/'separation_bundle.zip') as bundle:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        plan = json.loads(z.read('manifest.json'))
        assert plan==json.loads(bundle.read('probe/manifest.json'))
        validate_plan(plan)
        for name in ('script.py','run_separation_check.py','run_file_probe_experiment.py','fixed_music_probe.npz'):
            assert z.read(name)==bundle.read(name),name
        assert z.read('SHA256SUMS.txt')==bundle.read('model/SHA256SUMS.txt')
        assert hashlib.sha256(z.read('fixed_music_probe.npz')).hexdigest()==plan['probe_sha256']
        with np.load(io.BytesIO(z.read('fixed_music_probe.npz')),allow_pickle=False) as data:
            model={k:data[k].copy() for k in ('w','b','mean','scale')}
        assert all(np.isfinite(v).all() for v in model.values()) and np.all(model['scale']>0)
        runtime = json.loads(z.read('runtime.json'))
        assert runtime['forced_separation'] and runtime['probe_sha256']==plan['probe_sha256']
        config=runtime['config']
        indexed={r['ID']:r for r in plan['rows']}
        rows=list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
        assert len(rows)==len(indexed)==80 and {r['ID'] for r in rows}==set(indexed)
        expected_features={'features/'+identity+'_'+view+'.npz' for identity in indexed for view in VIEWS}
        assert {n for n in z.namelist() if n.startswith('features/') and n.endswith('.npz')}==expected_features
        assert {n for n in z.namelist() if n.startswith('music_stems/') and n.endswith('.wav')}=={'music_stems/'+identity+'.wav' for identity in indexed}
        maximum_error=0.
        for row in rows:
            for key in row:
                if key in ('ID','family','music_stem_sha256'):
                    continue
                if key in ('music_comparison_eligible','music_silent','production_gate_would_separate'):
                    assert row[key] in ('True','False')
                    row[key]=row[key]=='True'
                elif key in ('file_fake','original_samples','music_samples'):
                    row[key]=int(row[key])
                else:
                    row[key]=float(row[key])
                    assert np.isfinite(row[key]),key
            planned=indexed[row['ID']]
            for key in ('family','file_fake','music_comparison_eligible'):
                assert row[key]==planned[key]
            blob=z.read('music_stems/'+row['ID']+'.wav')
            assert hashlib.sha256(blob).hexdigest()==row['music_stem_sha256']
            sr,music=wavfile.read(io.BytesIO(blob))
            assert sr==16000 and music.ndim==1 and len(music)==row['music_samples']
            assert np.isfinite(music).all() and abs(len(music)-row['original_samples'])<=16
            original_blob=bundle.read('probe/test/'+row['ID']+'.wav')
            assert hashlib.sha256(original_blob).hexdigest()==planned['clip_sha256']
            sr,original=wavfile.read(io.BytesIO(original_blob))
            assert sr==16000 and len(original)==row['original_samples']==128000
            leveled,gain,silent=level_control(music,config['silence_rms'])
            np.testing.assert_allclose(row['level_gain'],gain,rtol=1e-12,atol=1e-12)
            assert row['music_silent']==silent
            for key,audio in [('music_rms',music),('original_rms',original)]:
                np.testing.assert_allclose(row[key],np.sqrt(np.mean(audio.astype(float)**2)),rtol=1e-12,atol=1e-12)
            for view,audio in zip(VIEWS,(original,music,leveled)):
                with np.load(io.BytesIO(z.read('features/'+row['ID']+'_'+view+'.npz')),allow_pickle=False) as features:
                    assert str(features['input_sha256'])==planned['clip_sha256']
                    embedding=features['embeddings']
                    assert embedding.ndim==2 and np.isfinite(embedding).all()
                    if not len(embedding):
                        assert np.sqrt(np.mean(audio.astype(float)**2))<config['silence_rms']
                        predicted=0.
                    else:
                        assert embedding.shape[1]==len(model['w'])
                        predicted=float(predict(model,embedding).max())
                    maximum_error=max(maximum_error,abs(row[view+'_probe']-predicted))
                    np.testing.assert_allclose(row[view+'_probe'],predicted,rtol=1e-12,atol=1e-12)
                    assert row[view+'_baseline']==float(features['baseline'])
                    assert all(0<=row[view+'_'+head]<=1 for head in ('probe','baseline'))
            reference=plan['reference_predictions'][row['ID']]
            assert row['previous_baseline_gap']==abs(row['original_baseline']-reference['baseline'])
            assert row['previous_probe_gap']==abs(row['original_probe']-reference['previous_music_probe'])
            assert row['production_gate_would_separate']==bool(not config['demucs_gating'] or
                (row['voice_present']>=config['gate_voice'] and row['music_present']>=config['gate_music']))
        report=summarize(rows)
        report['maximum_previous_original_gap']=max(max(r['previous_baseline_gap'],r['previous_probe_gap']) for r in rows)
        assert report==json.loads(z.read('summary.json'))
        assert 'Traceback (most recent call last)' not in z.read('run.log').decode()
        real_speech=[r for r in rows if r['family']=='speech' and r['file_fake']==0]
        audit=dict(result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            verified_inputs=len(rows),verified_features=len(expected_features),
            maximum_probe_reconstruction_error=maximum_error,metrics=report,
            real_speech_control=dict(n=len(real_speech),
                mean_music_raw_probe=float(np.mean([r['music_raw_probe'] for r in real_speech])),
                raw_probe_ge_0_5=sum(r['music_raw_probe']>=.5 for r in real_speech),
                gate_would_separate_count=sum(r['production_gate_would_separate'] for r in real_speech)),
            total_recorded_seconds=sum(r['seconds'] for r in rows),
            gpu_features_independently_recomputed=False,
            voice_stem_reconstruction_independently_verified=False)
    out=ROOT/'data/experiments/separation-20260916'
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
