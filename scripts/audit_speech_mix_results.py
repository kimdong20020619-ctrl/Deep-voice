"""Recompute returned metrics and validate the actual generated mixture inputs."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from scipy.io import wavfile
from run_file_probe_experiment import evaluate
from run_speech_mix_check import mix

ROOT = Path(__file__).resolve().parents[1]
source = Path.home() / 'Downloads/speech_mix_results.zip'
with zipfile.ZipFile(source) as z, zipfile.ZipFile(ROOT/'speech_mix_bundle.zip') as bundle:
    assert z.testzip() is None
    plan = json.loads(z.read('manifest.json'))
    assert plan == json.loads(bundle.read('probe/manifest.json'))
    inputs = json.loads(z.read('input_manifest.json'))
    predictions = list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
    assert len(inputs) == len(predictions) == 56
    assert len({r['ID'] for r in inputs}) == len({r['ID'] for r in predictions}) == 56
    by_id = {r['ID']: r for r in inputs}
    assert set(by_id) == {r['ID'] for r in predictions}
    waves = {}
    for row in inputs:
        blob = z.read('inputs/'+row['ID']+'.wav')
        assert hashlib.sha256(blob).hexdigest() == row['sha256']
        sr, audio = wavfile.read(io.BytesIO(blob))
        assert sr == 16000 and audio.shape == (128000,)
        assert np.isfinite(audio).all() and np.abs(audio).max() <= .950001
        waves[row['ID']] = audio
    max_mix_error = 0.0
    for i, music in enumerate(plan['music']):
        for db in (-6, 6):
            identity = f'X{i:02d}_{db:+d}'
            row = by_id[identity]
            assert row['file_fake'] == music['file_fake']
            assert row['source_ids'] == [plan['speech'][i % 8]['source'], music['source']]
            expected, _ = mix(waves[f'S{i%8:02d}'], waves[f'M{i:02d}'], db)
            error = float(np.abs(expected-waves[identity]).max())
            max_mix_error = max(max_mix_error, error)
            assert error < 2e-6, (identity, error)
    for row in predictions:
        assert int(row['file_fake']) == by_id[row['ID']]['file_fake']
        assert row['condition'] == by_id[row['ID']]['condition']
        for key in row:
            if key.endswith('_PROB'):
                value = float(row[key])
                assert np.isfinite(value) and 0 <= value <= 1
                if key.startswith('baseline_') and key != 'baseline_FILE_FAKE_PROB':
                    np.testing.assert_allclose(value, float(row[key.replace('baseline_', 'probe_')]), atol=1e-5, rtol=1e-5)
    report = {}
    supplied = json.loads(z.read('summary.json'))
    for condition in sorted({r['condition'] for r in predictions}):
        subset = [r for r in predictions if r['condition'] == condition]
        truth = [int(r['file_fake']) for r in subset]
        report[condition] = {}
        for prefix in ('baseline', 'probe'):
            scores = np.array([float(r[prefix+'_FILE_FAKE_PROB']) for r in subset])
            metric = evaluate(truth, scores) if len(set(truth)) == 2 else dict(n=len(truth), file_eer=None, file_auc=None)
            metric.update(fake_fraction_at_0_5=float(np.mean(scores >= .5)), mean_fake_probability=float(np.mean(scores)))
            assert metric == supplied[condition][prefix]
            report[condition][prefix] = metric
    assert 'Traceback (most recent call last)' not in z.read('run.log').decode()
    audit = dict(result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 verified_inputs=56, maximum_reconstructed_mix_error=max_mix_error,
                 total_pair_inference_seconds=sum(float(r['seconds']) for r in predictions), metrics=report)
out = ROOT / 'data/experiments/speech-mix-20260916'
out.mkdir(parents=True, exist_ok=True)
target = out/source.name
if target.exists():
    assert target.read_bytes() == source.read_bytes()
else:
    shutil.copy2(source, target)
(out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit, indent=2))
