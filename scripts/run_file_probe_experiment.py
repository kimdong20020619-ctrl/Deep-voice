"""GPU feature extraction and train-only FILE probe with sealed group holdout."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np


def validate_rows(rows):
    if len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs')
    splits = ('train', 'development', 'holdout')
    if any(r['split'] not in splits or r['file_fake'] not in (0, 1) for r in rows):
        raise ValueError('Invalid split or label')
    for name in splits:
        if {r['file_fake'] for r in rows if r['split'] == name} != {0, 1}:
            raise ValueError('Missing real/fake class in ' + name)
    for field in ('group', 'source_sha256', 'clip_sha256'):
        seen = {}
        for r in rows:
            key = (r['file_fake'], r[field]) if field == 'group' else r[field]
            if key in seen and seen[key] != r['split']:
                raise ValueError('Split leakage: ' + field)
            seen[key] = r['split']
    generator_splits = {}
    for row in rows:
        if row['file_fake']:
            generator = row['generator']
            if generator in generator_splits and generator_splits[generator] != row['split']:
                raise ValueError('Generator leakage')
            generator_splits[generator] = row['split']


def evaluate(truth, scores):
    from sklearn.metrics import roc_curve, roc_auc_score
    truth, scores = np.asarray(truth), np.asarray(scores)
    if set(truth.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Invalid evaluation population')
    fpr, tpr, _ = roc_curve(truth, scores, pos_label=1, drop_intermediate=False)
    fnr = 1-tpr
    index = np.argmin(np.abs(fpr-fnr))
    return {'file_eer': float((fpr[index]+fnr[index])/2), 'file_auc': float(roc_auc_score(truth, scores)), 'n': len(truth)}


def run(work):
    from scipy.io import wavfile
    import torch
    from sklearn.exceptions import ConvergenceWarning
    manifest = json.loads((work / 'probe/manifest.json').read_text(encoding='utf-8'))
    rows = manifest['rows']
    validate_rows(rows)
    result = work / 'probe_results'
    result.mkdir(exist_ok=True)
    if (result / 'holdout.json').exists():
        raise ValueError('Holdout already evaluated; keep the existing result')
    (result / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (result / 'baseline_script.py').write_bytes((work / 'script.py').read_bytes())
    (result / 'SHA256SUMS.txt').write_bytes((work / 'model/SHA256SUMS.txt').read_bytes())
    spec = importlib.util.spec_from_file_location('probe_inference', work / 'script.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.CONFIG['file_head'] == 'direct' and script.CONFIG['segment_agg'] == 'max'
    assert torch.cuda.is_available()
    sys.path.insert(0, str(work / 'src'))
    from train.fit_probe import fit, predict, save
    extraction_start = time.perf_counter()
    scorer = script.DFArenaScorer(torch.device('cuda'))
    assert scorer.has_embeddings
    cache = result / 'features'
    cache.mkdir(exist_ok=True)
    identity = hashlib.sha256((work / 'script.py').read_bytes() + (work / 'model/SHA256SUMS.txt').read_bytes()).hexdigest()
    config_path = result / 'extraction_identity.json'
    config = {'identity': identity, 'torch': torch.__version__, 'device': torch.cuda.get_device_name(0)}
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Do not reuse another environment cache'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    features, baseline = {}, {}
    for i, row in enumerate(rows):
        audio_path = work / 'probe/test' / (row['ID'] + '.wav')
        assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == row['clip_sha256']
        target = cache / (row['ID'] + '.npz')
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                assert str(data['clip_sha256']) == row['clip_sha256']
                score, emb = float(data['baseline']), data['embeddings']
        else:
            sr, pcm = wavfile.read(audio_path)
            assert sr == 16000 and pcm.dtype == np.int16 and pcm.shape == (128000,)
            score, emb = scorer.score(pcm.astype(np.float32)/32768, kind='file', want_embeddings=True)
            if emb is None:
                raise ValueError('Missing embeddings: ' + row['ID'])
            np.savez(target, baseline=score, embeddings=emb, clip_sha256=row['clip_sha256'])
        assert emb.ndim == 2 and len(emb) > 0 and np.isfinite(emb).all() and np.isfinite(score) and 0 <= score <= 1
        features[row['ID']], baseline[row['ID']] = emb, score
        print('features', i+1, '/', len(rows), row['ID'], flush=True)
    (result / 'extraction_seconds.txt').write_text(str(time.perf_counter()-extraction_start))
    del scorer
    torch.cuda.empty_cache()
    train = [r for r in rows if r['split'] == 'train']
    X = np.concatenate([features[r['ID']] for r in train])
    y = np.concatenate([np.full(len(features[r['ID']]), r['file_fake']) for r in train])
    candidates = []
    for C in (0.001, 0.01, 0.1):
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            probe = fit(X, y, C=C, seed=20260915)
        development = [r for r in rows if r['split'] == 'development']
        scores = [float(predict(probe, features[r['ID']]).max()) for r in development]
        metric = evaluate([r['file_fake'] for r in development], scores)
        candidates.append((metric['file_eer'], -metric['file_auc'], C, probe, metric))
    selected = min(candidates, key=lambda item: item[:3])
    chosen = selected[3]
    save(result / 'experimental_file_probe.npz', chosen)
    # Same numerical implementation as the submission class; not a full pipeline test.
    runtime_probe = script.LinearProbe(result / 'experimental_file_probe.npz')
    np.testing.assert_allclose(runtime_probe.predict(X), predict(chosen, X), atol=1e-12, rtol=1e-12)
    (result / 'selection.json').write_text(json.dumps({'selected_C': selected[2],
        'criterion': 'development EER, then AUC, then smaller C; no holdout tuning',
        'candidates': [{'C': item[2], **item[4]} for item in candidates]}, indent=2), encoding='utf-8')
    output = []
    report = {}
    for split in ('development', 'holdout'):
        subset = [r for r in rows if r['split'] == split]
        truth = [r['file_fake'] for r in subset]
        scores = [float(predict(chosen, features[r['ID']]).max()) for r in subset]
        report[split] = {'baseline': evaluate(truth, [baseline[r['ID']] for r in subset]), 'probe': evaluate(truth, scores)}
        output.extend(dict(ID=r['ID'], split=split, file_fake=r['file_fake'], baseline=baseline[r['ID']], probe=s) for r, s in zip(subset, scores))
    with (result / 'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    (result / 'holdout.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (result / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    print('Diagnostic only. No competition Score or automatic deployment.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    run(parser.parse_args().work.resolve())
