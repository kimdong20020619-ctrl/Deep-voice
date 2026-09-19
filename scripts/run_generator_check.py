"""Matched-input comparison; all samples are previously used development material."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from run_file_probe_experiment import evaluate
from run_sonics_check import windows, normalized


def file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def crop(audio, seconds):
    length=seconds*16000
    assert audio.ndim==1 and len(audio)==128000 and seconds in (5,8)
    start=(len(audio)-length)//2
    return audio[start:start+length].copy(),start


def summarize(rows):
    report={}
    generators=sorted({r['generator'] for r in rows if r['file_fake']==1})
    for seconds in (5,8):
        for view in ('original','mp3_64k'):
            condition=[r for r in rows if r['seconds_input']==seconds and r['view']==view]
            for generator in generators:
                group=[r for r in condition if r['file_fake']==0 or r['generator']==generator]
                assert {r['file_fake'] for r in group}=={0,1}
                report[f'{seconds}s/{view}/{generator}']={head:evaluate([r['file_fake'] for r in group],[r[head] for r in group])
                    for head in ('sonics_mean','df_arena')}
    return report


def run(work):
    import torch
    import librosa
    import timm
    from scipy.io import wavfile
    from sonics import HFAudioClassifier
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    plan=json.loads((work/'plan.json').read_text())
    result=work/'generator_results'
    result.mkdir(exist_ok=True)
    assert not (result/'summary.json').exists(), 'Completed result exists'
    for name in ('plan.json','script.py','run_generator_check.py','run_sonics_check.py','run_file_probe_experiment.py'):
        shutil.copy2(work/name,result/name)
    shutil.copytree(work/'provenance',result/'provenance',dirs_exist_ok=True)
    for name,sha in plan['file_sha256'].items():
        assert hashlib.sha256((work/name).read_bytes()).hexdigest()==sha,name
    weight_paths={'sonics':work/'weights/alpha-5s/pytorch_model.bin','df_arena':work/'model/df_arena_1b/pytorch_model.bin'}
    weight_checks={name:file_sha256(path) for name,path in weight_paths.items()}
    assert weight_checks==plan['weight_sha256']
    (result/'weight_checks.json').write_text(json.dumps(weight_checks,indent=2))
    (result/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,librosa=librosa.__version__,timm=timm.__version__,
        gpu=torch.cuda.get_device_name(0),fit=False),indent=2))
    cache=work/'matched_inputs'
    cache.mkdir(exist_ok=True)
    tasks=[]
    for row in plan['rows']:
        sr,audio=wavfile.read(work/row['path'])
        assert sr==16000 and audio.dtype==np.int16 and len(audio)==128000
        audio=audio.astype(np.float32)/32768.0
        for seconds in (5,8):
            cropped,start=crop(audio,seconds)
            stem=row['ID']+'_'+str(seconds)+'s'
            mp3=cache/(stem+'.mp3')
            subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-f','f32le','-ar','16000','-ac','1','-i','pipe:0',
                '-codec:a','libmp3lame','-b:a','64k',str(mp3)],input=cropped.astype('<f4').tobytes(),check=True,capture_output=True)
            raw=subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(mp3),'-f','f32le','-ar','16000','-ac','1','pipe:1'],check=True,capture_output=True).stdout
            decoded=np.frombuffer(raw,dtype='<f4').copy()
            assert len(decoded)==len(cropped) and np.isfinite(decoded).all(), 'Codec changed input duration'
            for view,values in (('original',cropped),('mp3_64k',decoded)):
                key=stem+'_'+view
                np.save(cache/(key+'.npy'),values)
                tasks.append(dict(row,key=key,seconds_input=seconds,view=view,crop_start_samples=start,
                    decoded_sha256=hashlib.sha256(values.astype('<f4').tobytes()).hexdigest()))
    (result/'inputs.json').write_text(json.dumps(tasks,indent=2))
    spec=importlib.util.spec_from_file_location('matched_inference',work/'script.py')
    script=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.CONFIG['file_head']=='direct' and script.resolve_agg('file')=='max'
    scorer=script.DFArenaScorer(torch.device('cuda'))
    scores={}
    for task in tasks:
        begin=time.perf_counter()
        audio=np.load(cache/(task['key']+'.npy'),allow_pickle=False)
        value=scorer.score(audio,kind='file')
        assert np.isfinite(value) and 0<=value<=1
        scores[task['key']]=dict(df_arena=float(value),df_seconds=time.perf_counter()-begin)
        print('DF',len(scores),'/',len(tasks),flush=True)
    (result/'df_predictions.json').write_text(json.dumps(scores,indent=2))
    del scorer
    torch.cuda.empty_cache()
    model=HFAudioClassifier.from_pretrained(str(work/'weights/alpha-5s'),map_location='cpu',strict=True).eval().cuda()
    model.requires_grad_(False)
    (result/'segments').mkdir(exist_ok=True)
    rows=[]
    fields=['ID','generator','historical_split','file_fake','seconds_input','view','sonics_mean','df_arena','sonics_seconds','df_seconds']
    with (result/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for task in tasks:
            begin=time.perf_counter()
            audio=np.load(cache/(task['key']+'.npy'),allow_pickle=False)
            clips=windows(audio,80000)
            logits=[]
            with torch.inference_mode():
                for start,clip in clips:
                    value=model(torch.from_numpy(normalized(clip)).unsqueeze(0).cuda()).float().reshape(-1)
                    assert value.numel()==1 and torch.isfinite(value).all()
                    logits.append(float(value.cpu()[0]))
            probabilities=1/(1+np.exp(-np.clip(np.array(logits,dtype=float),-700,700)))
            row={k:task[k] for k in ('ID','generator','historical_split','file_fake','seconds_input','view')}
            row.update(scores[task['key']],sonics_mean=float(probabilities.mean()),sonics_seconds=time.perf_counter()-begin)
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            np.savez(result/'segments'/(task['key']+'.npz'),logits=logits,starts=[s for s,_ in clips],decoded_sha256=task['decoded_sha256'])
            print('SONICS',len(rows),'/',len(tasks),flush=True)
    assert len(rows)==plan['expected_predictions']==336
    assert {n:file_sha256(p) for n,p in weight_paths.items()}==weight_checks
    summary=dict(metrics=summarize(rows),note='Repeated development data; shared REAL controls; no fitting, selection or competition Score.')
    (result/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    run(parser.parse_args().work.resolve())
