"""Frozen SONICS screening on existing public development audio; no fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from run_file_probe_experiment import evaluate


def windows(audio, length):
    if len(audio) <= length:
        starts = [0]
    else:
        starts = list(range(0, len(audio)-length+1, length))
        if starts[-1] != len(audio)-length:
            starts.append(len(audio)-length)
    return [(start, np.pad(audio[start:start+length], (0, max(0, start+length-len(audio))))) for start in starts]


def normalized(audio):
    return np.asarray(audio, dtype=np.float32) / np.maximum(np.std(audio), 1e-6)


def summarize(rows):
    report = {}
    for model in sorted({r['model'] for r in rows}):
        for view in ('original', 'mp3_64k'):
            for cohort in ('external', 'development_music', 'development_controls'):
                group = [r for r in rows if (r['model'],r['view'],r['cohort']) == (model,view,cohort)]
                if not group:
                    continue
                result = {head:evaluate([r['file_fake'] for r in group], [r[head] for r in group]) for head in ('mean_prob','max_prob')}
                if view == 'original':
                    result['reference_df_arena'] = evaluate([r['file_fake'] for r in group], [r['reference_baseline'] for r in group])
                report[f'{model}/{view}/{cohort}'] = result
    return report


def run(work):
    import librosa
    import torch
    import timm
    from sonics import HFAudioClassifier
    assert torch.cuda.is_available(), 'GPU required'
    torch.manual_seed(42)
    torch.set_num_threads(2)
    plan = json.loads((work/'plan.json').read_text(encoding='utf-8'))
    result = work/'sonics_results'
    result.mkdir(exist_ok=True)
    assert not (result/'summary.json').exists(), 'Completed result exists'
    for name in ('plan.json','run_sonics_check.py','run_file_probe_experiment.py'):
        shutil.copy2(work/name, result/name)
    shutil.copytree(work/'provenance', result/'provenance', dirs_exist_ok=True)
    for relative, sha in plan['file_sha256'].items():
        assert hashlib.sha256((work/relative).read_bytes()).hexdigest() == sha, relative
    (result/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,timm=timm.__version__,librosa=librosa.__version__,
        gpu=torch.cuda.get_device_name(0),fit=False,primary='alpha-5s mean probability',
        notes='120s is right-zero-padded diagnostic only; controls are task-mismatch diagnostics, not Music EER.'),indent=2),encoding='utf-8')
    cache = work/'decoded'
    cache.mkdir(exist_ok=True)
    transform_log = []
    for row in plan['rows']:
        audio,_ = librosa.load(work/row['path'], sr=16000, mono=True)
        assert audio.ndim == 1 and len(audio)>0 and np.isfinite(audio).all()
        np.save(cache/(row['ID']+'_original.npy'),audio)
        mp3 = cache/(row['ID']+'.mp3')
        subprocess.run(['ffmpeg','-v','error','-nostdin','-y','-f','f32le','-ar','16000','-ac','1','-i','pipe:0',
            '-codec:a','libmp3lame','-b:a','64k',str(mp3)],input=audio.astype('<f4').tobytes(),check=True,capture_output=True)
        raw = subprocess.run(['ffmpeg','-v','error','-nostdin','-i',str(mp3),'-f','f32le','-ar','16000','-ac','1','pipe:1'],
            check=True,capture_output=True).stdout
        decoded = np.frombuffer(raw,dtype='<f4').copy()
        assert np.isfinite(decoded).all() and abs(len(decoded)-len(audio))<=1600
        np.save(cache/(row['ID']+'_mp3_64k.npy'),decoded)
        transform_log.append(dict(ID=row['ID'],original_samples=len(audio),mp3_samples=len(decoded),
            mp3_sha256=hashlib.sha256(mp3.read_bytes()).hexdigest(),decoded_sha256=hashlib.sha256(raw).hexdigest()))
    (result/'transforms.json').write_text(json.dumps(transform_log,indent=2),encoding='utf-8')
    predictions = []
    fields = ['ID','cohort','family','file_fake','model','view','mean_prob','max_prob','reference_baseline','segments','seconds']
    with (result/'predictions.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for variant, model_plan in plan['models'].items():
            model_path=work/'weights'/variant
            assert hashlib.sha256((model_path/'pytorch_model.bin').read_bytes()).hexdigest()==model_plan['weights_sha256']
            assert hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest()==model_plan['config_sha256']
            model=HFAudioClassifier.from_pretrained(str(model_path),map_location='cpu',strict=True).eval().cuda()
            model.requires_grad_(False)
            length=80000 if variant=='alpha-5s' else 1920000
            for row in plan['rows']:
                if variant=='alpha-120s' and row['cohort']!='external':
                    continue
                for view in ('original','mp3_64k'):
                    begin=time.perf_counter()
                    audio=np.load(cache/(row['ID']+'_'+view+'.npy'),allow_pickle=False)
                    clips=windows(audio,length)
                    logits=[]
                    with torch.inference_mode():
                        for start, clip in clips:
                            x=torch.from_numpy(normalized(clip)).unsqueeze(0).cuda()
                            value=model(x).float().reshape(-1)
                            assert value.numel()==1 and torch.isfinite(value).all()
                            logits.append(float(value.cpu()[0]))
                    probabilities=1/(1+np.exp(-np.clip(np.array(logits,dtype=float),-700,700)))
                    item={k:row[k] for k in ('ID','cohort','family','file_fake','reference_baseline')}
                    item.update(model=variant,view=view,mean_prob=float(probabilities.mean()),max_prob=float(probabilities.max()),
                        segments=len(logits),seconds=time.perf_counter()-begin)
                    writer.writerow(item)
                    stream.flush()
                    predictions.append(item)
                    details=result/'segments'
                    details.mkdir(exist_ok=True)
                    np.savez(details/(row['ID']+'_'+variant+'_'+view+'.npz'),logits=logits,starts=[s for s,_ in clips],input_samples=len(audio))
                    print('done',len(predictions),'/',plan['expected_predictions'],row['ID'],variant,view,flush=True)
            del model
            torch.cuda.empty_cache()
    assert len(predictions)==plan['expected_predictions']
    for variant, entry in plan['models'].items():
        assert hashlib.sha256((work/'weights'/variant/'pytorch_model.bin').read_bytes()).hexdigest()==entry['weights_sha256']
    summary=dict(metrics=summarize(predictions),note='Screening only; no fitting, no automatic selection, no competition Score.')
    (result/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    run(parser.parse_args().work.resolve())
