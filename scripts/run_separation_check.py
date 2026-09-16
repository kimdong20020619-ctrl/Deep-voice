"""Fixed-weight separation diagnostic, with amplitude control and explicit populations."""
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

VIEWS = ('original', 'music_raw', 'music_level')


def level_control(audio, threshold):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError('Invalid separated audio')
    rms = float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
    if rms < threshold:
        return audio.copy(), 1.0, True
    gain = min(.05/rms, .95/max(float(np.abs(audio).max()), 1e-12))
    return (audio*gain).astype(np.float32), gain, False


def validate_plan(plan):
    rows = plan['rows']
    sources = {s['ID']:s for s in plan['sources']}
    if len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input IDs')
    for row in rows:
        selected = [sources[s] for s in row['source_ids']]
        if row['split'] != 'development' or any(s['split']!='development' for s in selected):
            raise ValueError('Development inputs only')
        if row['file_fake'] != max(s['file_fake'] for s in selected):
            raise ValueError('Incorrect FILE truth')
        eligible = any(s['kind']=='music' for s in selected) and not any(s['kind']=='speech' and s['file_fake'] for s in selected)
        if row['music_comparison_eligible'] != eligible:
            raise ValueError('Incorrect diagnostic population')


def summarize(rows):
    metrics = {}
    for family in sorted({r['family'] for r in rows if r['music_comparison_eligible']}):
        subset = [r for r in rows if r['family']==family and r['music_comparison_eligible']]
        truth = [r['file_fake'] for r in subset]
        metrics[family] = {view:{head:evaluate(truth,[r[view+'_'+head] for r in subset])
                                for head in ('baseline','probe')} for view in VIEWS}
    controls = {}
    for family in sorted({r['family'] for r in rows if not r['music_comparison_eligible']}):
        subset = [r for r in rows if r['family']==family and not r['music_comparison_eligible']]
        controls[family] = dict(n=len(subset), note='Descriptive scores only; not FILE/component EER',
            mean_music_raw_probe=float(np.mean([r['music_raw_probe'] for r in subset])),
            silent_music_count=sum(r['music_silent'] for r in subset))
    return dict(eligible_file_metrics=metrics, excluded_controls=controls,
                note='Development diagnostic. No Music EER, Voice EER, overall competition Score, or automatic model selection.')


def run(work):
    import torch
    from scipy.io import wavfile
    plan = json.loads((work/'probe/manifest.json').read_text(encoding='utf-8'))
    validate_plan(plan)
    probe_path = work/'fixed_music_probe.npz'
    assert hashlib.sha256(probe_path.read_bytes()).hexdigest() == plan['probe_sha256']
    result = work/'separation_results'
    result.mkdir(exist_ok=True)
    if (result/'summary.json').exists():
        raise ValueError('Completed diagnostic already exists; retain previous results')
    for name in ('features','music_stems','provenance'):
        (result/name).mkdir(exist_ok=True)
    for path in (work/'probe').glob('*.txt'):
        shutil.copy2(path,result/'provenance'/path.name)
    for name in ('script.py','run_separation_check.py','run_file_probe_experiment.py','fixed_music_probe.npz'):
        shutil.copy2(work/name,result/name)
    shutil.copy2(work/'model/SHA256SUMS.txt',result/'SHA256SUMS.txt')
    shutil.copy2(work/'probe/manifest.json',result/'manifest.json')
    spec = importlib.util.spec_from_file_location('separation_inference',work/'script.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.resolve_agg('file') == script.resolve_agg('music') == 'max'
    assert torch.cuda.is_available(), 'Select T4 GPU'
    device = torch.device('cuda')
    scorer = script.DFArenaScorer(device)
    assert scorer.has_embeddings
    demucs = script.load_htdemucs_model(device)
    panns = script.load_panns_model(device)
    probe = script.LinearProbe(probe_path)
    (result/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__,device=torch.cuda.get_device_name(0),
        config=script.CONFIG, forced_separation=True, probe_sha256=plan['probe_sha256']),indent=2),encoding='utf-8')
    rows = []
    columns = ['ID','family','file_fake','music_comparison_eligible','voice_present','music_present',
        'production_gate_would_separate','original_rms','music_rms','music_silent','level_gain',
        'original_samples','music_samples','reconstruction_relative_rms','music_stem_sha256','seconds']
    columns += [view+'_'+head for view in VIEWS for head in ('baseline','probe')]
    columns += ['previous_baseline_gap','previous_probe_gap']
    with (result/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for entry in plan['rows']:
            start = time.perf_counter()
            path = work/'probe/test'/(entry['ID']+'.wav')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['clip_sha256']
            audio = script.load_audio_16k(path)
            assert audio.shape == (128000,) and np.isfinite(audio).all()
            voice_present,music_present = script.predict_presence(panns,audio)
            assert all(np.isfinite(v) and 0<=v<=1 for v in (voice_present,music_present))
            voice,music = script.separate_voice_and_music(audio,demucs,device)
            for stem in (voice,music):
                assert stem.ndim==1 and np.isfinite(stem).all() and abs(len(stem)-len(audio))<=16
            assert len(voice)==len(music)
            leveled,gain,silent = level_control(music,script.silence_threshold())
            stem_path = result/'music_stems'/(entry['ID']+'.wav')
            wavfile.write(stem_path,16000,music.astype(np.float32))
            size = min(len(audio),len(music))
            residual = script.calculate_rms(audio[:size]-voice[:size]-music[:size])/max(script.calculate_rms(audio[:size]),1e-12)
            output = {k:entry[k] for k in ('ID','family','file_fake','music_comparison_eligible')}
            output.update(voice_present=voice_present,music_present=music_present,
                production_gate_would_separate=bool(not script.CONFIG['demucs_gating'] or
                    (voice_present>=script.CONFIG['gate_voice'] and music_present>=script.CONFIG['gate_music'])),
                original_rms=script.calculate_rms(audio),music_rms=script.calculate_rms(music),
                music_silent=bool(silent),level_gain=gain,original_samples=len(audio),music_samples=len(music),
                reconstruction_relative_rms=residual,music_stem_sha256=hashlib.sha256(stem_path.read_bytes()).hexdigest())
            for view, signal in zip(VIEWS,(audio,music,leveled)):
                baseline,embedding = scorer.score(signal,kind='file' if view=='original' else 'music',want_embeddings=True)
                if embedding is None:
                    assert script.calculate_rms(signal)<script.silence_threshold(), 'Non-silent input missing embeddings'
                    score = 0.0
                    embedding = np.empty((0,0),dtype=np.float32)
                else:
                    assert embedding.ndim==2 and len(embedding)>0 and np.isfinite(embedding).all()
                    score = float(probe.predict(embedding).max())
                assert all(np.isfinite(v) and 0<=v<=1 for v in (baseline,score))
                output[view+'_baseline'],output[view+'_probe'] = baseline,score
                np.savez(result/'features'/(entry['ID']+'_'+view+'.npz'),baseline=baseline,
                         embeddings=embedding,input_sha256=entry['clip_sha256'])
            reference = plan['reference_predictions'][entry['ID']]
            output['previous_baseline_gap'] = abs(output['original_baseline']-reference['baseline'])
            output['previous_probe_gap'] = abs(output['original_probe']-reference['previous_music_probe'])
            output['seconds'] = time.perf_counter()-start
            writer.writerow(output)
            f.flush()
            rows.append(output)
            print('done',len(rows),'/',len(plan['rows']),entry['ID'],
                  'probe original/raw/level',*[output[v+'_probe'] for v in VIEWS],flush=True)
    assert hashlib.sha256(probe_path.read_bytes()).hexdigest() == plan['probe_sha256']
    report = summarize(rows)
    report['maximum_previous_original_gap'] = max(max(r['previous_baseline_gap'],r['previous_probe_gap']) for r in rows)
    (result/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    run(parser.parse_args().work.resolve())
