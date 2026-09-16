"""Frozen probe amplitude and real-pipeline diagnostic; no fitting or holdout tuning."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from run_file_probe_experiment import evaluate


def run(work):
    import torch
    manifest = json.loads((work / 'probe/manifest.json').read_text())
    probe_path = work / 'frozen_probe.npz'
    assert hashlib.sha256(probe_path.read_bytes()).hexdigest() == manifest['probe_sha256']
    result = work / 'frozen_probe_results'
    result.mkdir(exist_ok=True)
    (result / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    spec = importlib.util.spec_from_file_location('frozen_inference', work / 'script.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.CONFIG['file_head'] == 'direct'
    script.CONFIG['file_probe_blend'] = 1.0
    device = torch.device('cuda')
    assert torch.cuda.is_available()
    panns = script.load_panns_model(device)
    scorer = script.DFArenaScorer(device)
    demucs = script.load_htdemucs_model(device)
    probe = script.LinearProbe(probe_path)
    columns = ['ID', 'condition', 'file_fake', 'baseline', 'probe', 'standalone_probe', 'pipeline_gap', 'seconds']
    predictions = []
    with (result / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in manifest['rows']:
            assert row['split'] == 'development', 'Do not reuse holdout for this diagnostic'
            path = work / 'probe/test' / (row['ID'] + '.wav')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['clip_sha256']
            started = time.perf_counter()
            baseline = script.process_one_file(path, panns, scorer, demucs, device)
            prediction = script.process_one_file(path, panns, scorer, demucs, device, file_probe=probe)
            audio = script.load_audio_16k(path)
            _, emb = scorer.score(audio, want_embeddings=True)
            assert emb is not None and np.isfinite(emb).all()
            standalone = script.aggregate_segment_scores(probe.predict(emb).tolist(), script.resolve_agg())
            gap = abs(standalone-prediction['FILE_FAKE_PROB'])
            for name, value in prediction.items():
                assert np.isfinite(value) and 0 <= value <= 1, name
                if name != 'FILE_FAKE_PROB':
                    np.testing.assert_allclose(value, baseline[name], atol=1e-5, rtol=1e-5)
            entry = dict(ID=row['ID'], condition=row['condition'], file_fake=row['file_fake'],
                         baseline=baseline['FILE_FAKE_PROB'], probe=prediction['FILE_FAKE_PROB'],
                         standalone_probe=standalone, pipeline_gap=gap, seconds=time.perf_counter()-started)
            writer.writerow(entry)
            f.flush()
            predictions.append(entry)
            print(row['ID'], 'pipeline gap', gap, flush=True)
            np.testing.assert_allclose(prediction['FILE_FAKE_PROB'], standalone, atol=1e-5, rtol=1e-5)
    summary = {}
    for condition in sorted({r['condition'] for r in predictions}):
        subset = [r for r in predictions if r['condition'] == condition]
        truth = [r['file_fake'] for r in subset]
        summary[condition] = {name: evaluate(truth, [r[name] for r in subset]) for name in ('baseline', 'probe')}
    assert hashlib.sha256(probe_path.read_bytes()).hexdigest() == manifest['probe_sha256']
    (result / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print('Development robustness only; not independent validation or competition Score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    run(parser.parse_args().work.resolve())
