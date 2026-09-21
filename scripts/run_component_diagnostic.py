"""Run frozen production functions on reviewed clips; no tuning or fallback scores."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

OUTPUTS = dict(file_fake='FILE_FAKE_PROB', voice_fake='VOICE_FAKE_PROB',
               music_fake='MUSIC_FAKE_PROB', voice_presence='VOICE_PRESENT_PROB',
               music_presence='MUSIC_PRESENT_PROB')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def metrics(rows, predictions):
    assert len(rows) == len(predictions)
    assert len({p['ID'] for p in predictions}) == len(predictions)
    indexed = {p['ID']: p for p in predictions}
    assert set(indexed) == {r['ID'] for r in rows}
    report = {}
    for label, column in OUTPUTS.items():
        selected = [r for r in rows if label not in ('voice_fake', 'music_fake')
                    or r[label.split('_')[0]+'_presence'] == 1]
        y = np.asarray([r[label] for r in selected], dtype=float)
        score = np.asarray([indexed[r['ID']][column] for r in selected], dtype=float)
        assert set(y).issubset({0, 1}) and np.isfinite(score).all()
        assert ((score >= 0) & (score <= 1)).all()
        entry = dict(n=len(y), positive=int(y.sum()), negative=int(len(y)-y.sum()),
                     IDs=[r['ID'] for r in selected])
        if len(set(y)) < 2:
            entry.update(auc=None, eer=None, reason='single_class_or_empty')
        else:
            entry['auc'] = float(roc_auc_score(y, score))
            if label.endswith('_fake'):
                fpr, tpr, _ = roc_curve(y, score, pos_label=1, drop_intermediate=False)
                fnr = 1-tpr
                i = np.argmin(np.abs(fpr-fnr))
                entry['eer'] = float((fpr[i]+fnr[i])/2)
        report[label] = entry
    return report


def run(work):
    import torch
    plan = json.loads((work/'plan.json').read_text(encoding='utf-8'))
    result = work/'component_results'
    result.mkdir(exist_ok=True)
    assert not (result/'predictions.csv').exists(), 'Use a fresh work directory for a rerun'
    for name, digest in plan['file_sha256'].items():
        assert sha(work/name) == digest, name
    weights = {}
    for line in (work/'model/SHA256SUMS.txt').read_text().splitlines():
        digest, relative = line.split()
        assert sha(work/'model'/relative) == digest, relative
        weights[relative] = digest
    for name in ('plan.json', 'script.py', 'run_component_diagnostic.py'):
        shutil.copy2(work/name, result/name)
    spec = importlib.util.spec_from_file_location('component_inference', work/'script.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert script.CONFIG == plan['config'], 'Configuration changed'
    assert torch.cuda.is_available(), 'Select T4 GPU runtime'
    device = torch.device('cuda')
    torch.manual_seed(20260920)
    np.random.seed(20260920)
    torch.backends.cudnn.benchmark = False
    original_load = torch.load
    # This legacy checkpoint was verified against the frozen SHA-256 above.
    def trusted_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)
    torch.load = trusted_load
    try:
        panns = script.load_panns_model(device)
    finally:
        torch.load = original_load
    scorer = script.DFArenaScorer(device)
    demucs = script.load_htdemucs_model(device)
    assert scorer.fake_index == scorer.model.config.label2id['spoof']
    runtime = dict(torch=torch.__version__, device=torch.cuda.get_device_name(0), config=script.CONFIG,
                   model_fake_index=scorer.fake_index, weights_sha256=weights,
                   fallback_predictions_allowed=False, panns_legacy_load_weights_only=False,
                   note='Current local process_one_file function; not proof of identity with historical leaderboard ZIP.')
    (result/'runtime.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    predictions = []
    with (result/'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['ID', *OUTPUTS.values(), 'seconds'])
        writer.writeheader()
        for row in plan['rows']:
            start = time.perf_counter()
            with torch.inference_mode():
                output = script.process_one_file(work/row['clip_path'], panns, scorer, demucs, device)
            assert set(output) == set(OUTPUTS.values())
            assert all(np.isfinite(v) and 0 <= v <= 1 for v in output.values())
            record = dict(ID=row['ID'], **{k:float(v) for k,v in output.items()}, seconds=time.perf_counter()-start)
            writer.writerow(record)
            stream.flush()
            predictions.append(record)
            print('done', len(predictions), '/', len(plan['rows']), row['ID'], flush=True)
    for relative, digest in weights.items():
        assert sha(work/'model'/relative) == digest
    report = dict(completed=True, predictions=len(predictions), metrics=metrics(plan['rows'], predictions),
                  interpretation='Tiny reused-source diagnostic. No competition Score, tuning, or improvement claim.')
    (result/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('COMPLETE', len(predictions), 'files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    run(parser.parse_args().work.resolve())
