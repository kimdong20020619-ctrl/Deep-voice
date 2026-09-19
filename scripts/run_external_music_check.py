"""Evaluate frozen FILE classifiers on new matched external music pairs; never fit."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from run_file_probe_experiment import evaluate


def validate_plan(plan):
    rows=plan['rows']
    if not rows or len({r['ID'] for r in rows})!=len(rows):
        raise ValueError('Empty or duplicate input IDs')
    groups={}
    hashes=set()
    for row in rows:
        if row['file_fake'] not in (0,1) or row['split']!='external_check':
            raise ValueError('Invalid external labels or split')
        if row['music_seconds']<8:
            raise ValueError('Insufficient annotated music')
        if row['clip_sha256'] in hashes:
            raise ValueError('Duplicate waveform')
        hashes.add(row['clip_sha256'])
        groups.setdefault(row['pair_id'],[]).append(row)
    used=set()
    used_tokens=set()
    used_speech=set()
    for group in groups.values():
        if len(group)!=2 or {r['file_fake'] for r in group}!={0,1}:
            raise ValueError('Unpaired class labels')
        if len({r['music_source_group'] for r in group})!=1:
            raise ValueError('Unmatched music origin')
        source=group[0]['music_source_group']
        if source in used:
            raise ValueError('Repeated music source across pairs')
        used.add(source)
        tokens=set(group[0]['music_source_tokens'])
        speech=set(group[0]['speech_source_ids'])
        if (group[0]['music_source_tokens']!=group[1]['music_source_tokens'] or
            group[0]['speech_source_ids']!=group[1]['speech_source_ids']):
            raise ValueError('Pair source metadata mismatch')
        if tokens & used_tokens or speech & used_speech:
            raise ValueError('Shared source across pairs')
        used_tokens.update(tokens)
        used_speech.update(speech)


def summarize(rows):
    report={}
    for head in ('baseline','music_probe','mixed_probe'):
        report[head]=evaluate([r['file_fake'] for r in rows],[r[head] for r in rows])
        pairs={}
        for row in rows:
            pairs.setdefault(row['pair_id'],{})[row['file_fake']]=row[head]
        report[head]['paired_fake_ranked_higher']=sum(p[1]>p[0] for p in pairs.values())
        report[head]['paired_ties']=sum(p[1]==p[0] for p in pairs.values())
        report[head]['pair_count']=len(pairs)
    return report


def run(work):
    import torch
    plan=json.loads((work/'probe/manifest.json').read_text(encoding='utf-8'))
    validate_plan(plan)
    result=work/'external_music_results'
    result.mkdir(exist_ok=True)
    if (result/'summary.json').exists():
        raise ValueError('Completed external evaluation exists; do not overwrite')
    (result/'features').mkdir(exist_ok=True)
    (result/'provenance').mkdir(exist_ok=True)
    for path in (work/'probe').glob('*.txt'):
        shutil.copy2(path,result/'provenance'/path.name)
    for name in ('record.json','selection_rejections.json'):
        if (work/'probe'/name).exists():
            shutil.copy2(work/'probe'/name,result/'provenance'/name)
    if (work/'probe/metadata').exists():
        shutil.copytree(work/'probe/metadata',result/'provenance/metadata',dirs_exist_ok=True)
    shutil.copy2(work/'probe/manifest.json',result/'manifest.json')
    shutil.copy2(work/'model/SHA256SUMS.txt',result/'SHA256SUMS.txt')
    for name in ('script.py','run_external_music_check.py','run_file_probe_experiment.py'):
        shutil.copy2(work/name,result/name)
    spec=importlib.util.spec_from_file_location('external_inference',work/'script.py')
    script=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.CONFIG['file_head']=='direct' and script.resolve_agg('file')=='max'
    models={}
    for name,sha in plan['model_sha256'].items():
        path=work/(name+'.npz')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
        models[name]=script.LinearProbe(path)
        shutil.copy2(path,result/path.name)
    assert set(models)=={'music_probe','mixed_probe'}
    assert torch.cuda.is_available(), 'Select GPU runtime'
    scorer=script.DFArenaScorer(torch.device('cuda'))
    assert scorer.has_embeddings
    (result/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),
        config=script.CONFIG,fit=False,model_sha256=plan['model_sha256']),indent=2),encoding='utf-8')
    predictions=[]
    with (result/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ID','pair_id','file_fake','baseline','music_probe','mixed_probe','samples_16k','seconds'])
        writer.writeheader()
        for row in plan['rows']:
            start=time.perf_counter()
            path=work/'probe/test'/(row['ID']+'.wav')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['clip_sha256']
            audio=script.load_audio_16k(path)
            assert audio.ndim==1 and np.isfinite(audio).all() and 59*16000<=len(audio)<=61*16000
            baseline,embedding=scorer.score(audio,kind='file',want_embeddings=True)
            assert embedding is not None and embedding.ndim==2 and len(embedding)>0 and np.isfinite(embedding).all()
            output={k:row[k] for k in ('ID','pair_id','file_fake')}
            output.update(baseline=float(baseline),samples_16k=len(audio))
            for name,model in models.items():
                output[name]=float(model.predict(embedding).max())
            assert all(np.isfinite(output[k]) and 0<=output[k]<=1 for k in ('baseline',*models))
            np.savez(result/'features'/(row['ID']+'.npz'),baseline=baseline,embeddings=embedding,clip_sha256=row['clip_sha256'])
            output['seconds']=time.perf_counter()-start
            writer.writerow(output)
            f.flush()
            predictions.append(output)
            print('done',len(predictions),'/',len(plan['rows']),row['ID'],flush=True)
    for name,sha in plan['model_sha256'].items():
        assert hashlib.sha256((work/(name+'.npz')).read_bytes()).hexdigest()==sha
    summary=dict(file_metrics=summarize(predictions),
        note='Frozen external source check only. No fitting, no component metrics, no competition Score or automatic deployment.')
    (result/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    run(parser.parse_args().work.resolve())
