"""Frozen FILE-probe regression check on real speech and controlled music mixtures."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from run_file_probe_experiment import evaluate


def unit_rms(audio):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.shape != (128000,) or not np.isfinite(audio).all():
        raise ValueError('Invalid eight-second mono audio')
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < 1e-6:
        raise ValueError('Silent source crop')
    return audio / rms * 0.05


def mix(voice, music, voice_to_music_db):
    voice = unit_rms(voice) * 10**(voice_to_music_db/20)
    music = unit_rms(music)
    audio = voice + music
    scale = min(1.0, 0.95/max(float(np.max(np.abs(audio))), 1e-12))
    return (audio*scale).astype(np.float32), scale


def safe_level(audio):
    audio = unit_rms(audio)
    return (audio * min(1.0, 0.95/max(float(np.max(np.abs(audio))), 1e-12))).astype(np.float32)


def run(work):
    import soundfile as sf
    import torch
    plan = json.loads((work / 'probe/manifest.json').read_text())
    result = work / 'speech_mix_results'
    result.mkdir(exist_ok=True)
    data = result / 'inputs'
    data.mkdir(exist_ok=True)
    (result / 'manifest.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    probe_path = work / 'frozen_probe.npz'
    assert hashlib.sha256(probe_path.read_bytes()).hexdigest() == plan['probe_sha256']
    speech, music = [], []
    for row in plan['speech']:
        path = work / 'probe/speech' / row['local_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        a, sr = sf.read(path, dtype='float32')
        assert sr == 16000 and a.ndim == 1 and len(a) == row['samples']
        start = (len(a)-128000)//2
        a = a[start:start+128000]
        unit_rms(a)
        speech.append((row, a))
    for row in plan['music']:
        path = work / 'probe/test' / (row['ID']+'.wav')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['clip_sha256']
        a, sr = sf.read(path, dtype='float32')
        assert sr == 16000
        unit_rms(a)
        music.append((row, a))
    rows = []

    def add(identity, condition, truth, audio, source_ids, **extra):
        path = data / (identity+'.wav')
        assert np.isfinite(audio).all() and np.max(np.abs(audio)) <= 0.950001
        sf.write(path, audio, 16000, subtype='FLOAT')
        rows.append(dict(ID=identity, condition=condition, file_fake=truth,
                         source_ids=source_ids, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra))

    for i, (row, audio) in enumerate(speech):
        add(f'S{i:02d}', 'speech_real', 0, safe_level(audio), [row['source']])
    for i, (row, audio) in enumerate(music):
        add(f'M{i:02d}', 'music_control', row['file_fake'], safe_level(audio), [row['source']])
        voice_row, voice = speech[i % len(speech)]
        for db in (-6, 6):
            combined, scale = mix(voice, audio, db)
            add(f'X{i:02d}_{db:+d}', f'mix_voice_{db:+d}dB', row['file_fake'], combined,
                [voice_row['source'], row['source']], shared_scale=scale, voice_to_music_db=db)
    assert len(rows) == 56
    (result / 'input_manifest.json').write_text(json.dumps(rows, indent=2))
    spec = importlib.util.spec_from_file_location('mix_inference', work / 'script.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.CONFIG['file_head'] == 'direct'
    script.CONFIG['file_probe_blend'] = 1.0
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    panns, scorer, demucs = script.load_panns_model(device), script.DFArenaScorer(device), script.load_htdemucs_model(device)
    probe = script.LinearProbe(probe_path)
    names = ['FILE_FAKE_PROB','VOICE_FAKE_PROB','MUSIC_FAKE_PROB','VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB']
    columns = ['ID','condition','file_fake','seconds'] + [prefix+n for prefix in ('baseline_', 'probe_') for n in names]
    values = []
    with (result / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            started = time.perf_counter()
            path = data / (row['ID']+'.wav')
            baseline = script.process_one_file(path, panns, scorer, demucs, device)
            candidate = script.process_one_file(path, panns, scorer, demucs, device, file_probe=probe)
            output = {k:row[k] for k in ('ID','condition','file_fake')}
            output['seconds'] = time.perf_counter()-started
            for prefix, predictions in (('baseline_', baseline), ('probe_', candidate)):
                for name in names:
                    value = predictions[name]
                    assert np.isfinite(value) and 0 <= value <= 1
                    output[prefix+name] = value
            writer.writerow(output)
            f.flush()
            values.append(output)
            for name in names[1:]:
                np.testing.assert_allclose(candidate[name], baseline[name], atol=1e-5, rtol=1e-5)
            print('done', row['ID'], 'baseline', baseline[names[0]], 'probe', candidate[names[0]], flush=True)
    report = {}
    for condition in sorted({r['condition'] for r in values}):
        subset = [r for r in values if r['condition']==condition]
        truth = [r['file_fake'] for r in subset]
        report[condition] = {}
        for prefix in ('baseline', 'probe'):
            scores = [r[prefix+'_FILE_FAKE_PROB'] for r in subset]
            metric = evaluate(truth,scores) if len(set(truth))==2 else {'n':len(truth),'file_eer':None,'file_auc':None}
            metric['fake_fraction_at_0_5'] = float(np.mean(np.asarray(scores)>=0.5))
            metric['mean_fake_probability'] = float(np.mean(scores))
            report[condition][prefix] = metric
    assert hashlib.sha256(probe_path.read_bytes()).hexdigest() == plan['probe_sha256']
    (result / 'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    print('No fake speech included: no Voice EER, component labels or competition Score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    run(parser.parse_args().work.resolve())
