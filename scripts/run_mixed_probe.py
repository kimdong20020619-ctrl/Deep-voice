"""Train a FILE probe on mixed inputs; development results are not a final holdout."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

import numpy as np
from run_file_probe_experiment import evaluate


def validate_manifest(manifest):
    rows, sources = manifest['rows'], manifest['sources']
    index = {s['ID']: s for s in sources}
    if len(index) != len(sources) or len({r['ID'] for r in rows}) != len(rows):
        raise ValueError('Duplicate identity')
    seen = {}
    for source in sources:
        if source['split'] not in ('train', 'development') or source['file_fake'] not in (0, 1):
            raise ValueError('Invalid source split or label')
        keys = [('group', source['source_group']), ('sha256', source['source_sha256'])]
        if source['file_fake']:
            keys.append(('generator', source['kind']+':'+source['generator']))
        for key in keys:
            if key in seen and seen[key] != source['split']:
                raise ValueError('Source leakage: '+str(key))
            seen[key] = source['split']
    clip_splits = {}
    for row in rows:
        selected = [index[identity] for identity in row['source_ids']]
        if not selected or any(s['split'] != row['split'] for s in selected):
            raise ValueError('Mixture source split mismatch')
        if row['file_fake'] != max(s['file_fake'] for s in selected):
            raise ValueError('Incorrect FILE truth')
        for field, source_field in (('source_groups', 'source_group'), ('source_hashes', 'source_sha256')):
            if row[field] != [s[source_field] for s in selected]:
                raise ValueError('Incorrect provenance')
        if any(s['kind']=='speech' and s['file_fake'] for s in selected) and len(selected)>1:
            music = next(s for s in selected if s['kind']=='music')
            if music['file_fake'] or not music.get('sa_compatible'):
                raise ValueError('Unapproved license combination')
        sha = row['clip_sha256']
        if sha in clip_splits and clip_splits[sha] != row['split']:
            raise ValueError('Derived clip leakage')
        clip_splits[sha] = row['split']
    for split in ('train', 'development'):
        for family in ('speech', 'music', 'mix_voice_-6dB', 'mix_voice_+6dB'):
            if {r['file_fake'] for r in rows if r['split']==split and r['family']==family} != {0, 1}:
                raise ValueError('Missing family classes: '+split+'/'+family)


def family_metrics(rows, scores):
    result = {}
    for family in sorted({r['family'] for r in rows}):
        subset = [r for r in rows if r['family']==family]
        result[family] = evaluate([r['file_fake'] for r in subset], [scores[r['ID']] for r in subset])
    result['all'] = evaluate([r['file_fake'] for r in rows], [scores[r['ID']] for r in rows])
    result['macro_family_eer'] = float(np.mean([result[f]['file_eer'] for f in result if f != 'all']))
    return result


def run(work):
    from scipy.io import wavfile
    import torch
    from sklearn.exceptions import ConvergenceWarning
    manifest = json.loads((work/'probe/manifest.json').read_text(encoding='utf-8'))
    validate_manifest(manifest)
    rows = manifest['rows']
    result = work/'mixed_probe_results'
    result.mkdir(exist_ok=True)
    if (result/'selection.json').exists():
        raise ValueError('Experiment already selected; keep existing results')
    for name in ('script.py', 'run_mixed_probe.py', 'run_file_probe_experiment.py', 'previous_music_probe.npz'):
        shutil.copy2(work/name, result/name)
    shutil.copy2(work/'probe/manifest.json', result/'manifest.json')
    shutil.copy2(work/'model/SHA256SUMS.txt', result/'SHA256SUMS.txt')
    shutil.copy2(work/'src/train/fit_probe.py', result/'fit_probe.py')
    code_hash = hashlib.sha256(b''.join((work/name).read_bytes() for name in
        ('script.py','run_mixed_probe.py','src/train/fit_probe.py','probe/manifest.json','model/SHA256SUMS.txt'))).hexdigest()
    spec = importlib.util.spec_from_file_location('mixed_inference', work/'script.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.CONFIG['file_head'] == 'direct' and script.CONFIG['segment_agg'] == 'max'
    assert torch.cuda.is_available(), 'Select GPU runtime'
    identity = dict(code_manifest_model_hash=code_hash, torch=torch.__version__, device=torch.cuda.get_device_name(0))
    cache_identity = result/'extraction_identity.json'
    if cache_identity.exists():
        assert json.loads(cache_identity.read_text()) == identity
    cache_identity.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    features, baseline, previous = {}, {}, {}
    cache = result/'features'
    cache.mkdir(exist_ok=True)
    old = script.LinearProbe(work/'previous_music_probe.npz')
    scorer = script.DFArenaScorer(torch.device('cuda'))
    assert scorer.has_embeddings
    started = time.perf_counter()
    for i, row in enumerate(rows):
        audio_path = work/'probe/test'/(row['ID']+'.wav')
        assert hashlib.sha256(audio_path.read_bytes()).hexdigest() == row['clip_sha256']
        target = cache/(row['ID']+'.npz')
        if target.exists():
            with np.load(target, allow_pickle=False) as data:
                assert str(data['clip_sha256']) == row['clip_sha256']
                score, embedding = float(data['baseline']), data['embeddings']
        else:
            sr, audio = wavfile.read(audio_path)
            assert sr == 16000 and audio.dtype == np.float32 and audio.shape == (128000,)
            assert np.isfinite(audio).all() and np.abs(audio).max() <= .950001
            score, embedding = scorer.score(audio, kind='file', want_embeddings=True)
            assert embedding is not None
            np.savez(target, baseline=score, embeddings=embedding, clip_sha256=row['clip_sha256'])
        assert embedding.ndim == 2 and len(embedding) > 0 and np.isfinite(embedding).all()
        assert np.isfinite(score) and 0 <= score <= 1
        features[row['ID']], baseline[row['ID']] = embedding, score
        previous[row['ID']] = float(old.predict(embedding).max())
        print('features',i+1,'/',len(rows),row['ID'],flush=True)
    (result/'extraction_seconds.txt').write_text(str(time.perf_counter()-started))
    del scorer
    torch.cuda.empty_cache()
    sys.path.insert(0, str(work/'src'))
    from train.fit_probe import fit, predict, save
    train = [r for r in rows if r['split']=='train']
    dev = [r for r in rows if r['split']=='development']
    X = np.concatenate([features[r['ID']] for r in train])
    y = np.concatenate([np.full(len(features[r['ID']]),r['file_fake']) for r in train])
    candidates = []
    for C in (.001, .01, .1):
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            probe = fit(X, y, C=C, seed=20260916)
        scores = {r['ID']:float(predict(probe, features[r['ID']]).max()) for r in dev}
        metric = family_metrics(dev, scores)
        candidates.append((metric['macro_family_eer'], metric['all']['file_eer'], C, probe, scores, metric))
    selected = min(candidates, key=lambda x:x[:3])
    save(result/'experimental_mixed_probe.npz', selected[3])
    runtime = script.LinearProbe(result/'experimental_mixed_probe.npz')
    np.testing.assert_allclose(runtime.predict(X), predict(selected[3], X), atol=1e-12, rtol=1e-12)
    report = dict(baseline=family_metrics(dev, baseline), previous_music_probe=family_metrics(dev, previous),
                  mixed_probe=selected[5], status='development_only_not_submission_ready')
    with (result/'predictions.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ID','family','file_fake','baseline','previous_music_probe','mixed_probe'])
        writer.writeheader()
        for row in dev:
            writer.writerow(dict(ID=row['ID'],family=row['family'],file_fake=row['file_fake'],
                baseline=baseline[row['ID']],previous_music_probe=previous[row['ID']],mixed_probe=selected[4][row['ID']]))
    (result/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (result/'selection.json').write_text(json.dumps(dict(selected_C=selected[2],
        criterion='development equal-family mean EER, then pooled EER, then smaller C',
        candidates=[dict(C=c[2],metrics=c[5]) for c in candidates]),indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    print('No holdout used. No automatic deployment. Full inference and competition Score unverified.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    run(parser.parse_args().work.resolve())
