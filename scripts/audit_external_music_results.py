"""Verify returned frozen external predictions against archived inputs and weights."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from run_external_music_check import validate_plan, summarize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from train.fit_probe import predict


def main():
    source = Path.home()/'Downloads/external_music_results.zip'
    with zipfile.ZipFile(source) as z, zipfile.ZipFile(ROOT/'external_music_bundle.zip') as bundle:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        plan = json.loads(z.read('manifest.json'))
        assert plan == json.loads(bundle.read('probe/manifest.json'))
        validate_plan(plan)
        for name in ('script.py', 'run_external_music_check.py', 'run_file_probe_experiment.py'):
            assert z.read(name) == bundle.read(name), name
        assert z.read('SHA256SUMS.txt') == bundle.read('model/SHA256SUMS.txt')
        for name in ('record.json', 'selection_rejections.json', 'ATTRIBUTION.txt'):
            assert z.read('provenance/'+name) == bundle.read('probe/'+name)
        models = {}
        for name, sha in plan['model_sha256'].items():
            blob = z.read(name+'.npz')
            assert blob == bundle.read(name+'.npz')
            assert hashlib.sha256(blob).hexdigest() == sha
            with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                models[name] = {k: data[k].copy() for k in ('w', 'b', 'mean', 'scale')}
            assert all(np.isfinite(v).all() for v in models[name].values())
            assert np.all(models[name]['scale'] > 0)
        runtime = json.loads(z.read('runtime.json'))
        assert runtime['fit'] is False and runtime['model_sha256'] == plan['model_sha256']
        assert runtime['config']['file_head'] == 'direct'
        assert runtime['config']['segment_agg'] == 'max'
        indexed = {r['ID']: r for r in plan['rows']}
        rows = list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
        assert len(rows) == len(indexed) == 48
        assert {r['ID'] for r in rows} == set(indexed)
        assert {n for n in z.namelist() if n.startswith('features/') and n.endswith('.npz')} == {'features/'+i+'.npz' for i in indexed}
        maximum_error = 0.
        segments = []
        for row in rows:
            for key in ('file_fake', 'samples_16k'):
                row[key] = int(row[key])
            for key in ('baseline', 'music_probe', 'mixed_probe', 'seconds'):
                row[key] = float(row[key])
                assert np.isfinite(row[key])
            assert row['seconds'] >= 0 and row['samples_16k'] == 960000
            original = indexed[row['ID']]
            assert row['file_fake'] == original['file_fake'] and row['pair_id'] == original['pair_id']
            assert hashlib.sha256(bundle.read('probe/test/'+row['ID']+'.wav')).hexdigest() == original['clip_sha256']
            metadata = z.read('provenance/metadata/'+row['ID']+'.json')
            assert metadata == bundle.read('probe/metadata/'+row['ID']+'.json')
            assert hashlib.sha256(metadata).hexdigest() == original['metadata_sha256']
            with np.load(io.BytesIO(z.read('features/'+row['ID']+'.npz')), allow_pickle=False) as data:
                assert str(data['clip_sha256']) == original['clip_sha256']
                embedding = data['embeddings']
                assert embedding.ndim == 2 and len(embedding) > 0 and np.isfinite(embedding).all()
                segments.append(len(embedding))
                assert float(data['baseline']) == row['baseline']
                for name, model in models.items():
                    assert embedding.shape[1] == len(model['w'])
                    predicted = float(predict(model, embedding).max())
                    maximum_error = max(maximum_error, abs(predicted-row[name]))
                    np.testing.assert_allclose(predicted, row[name], rtol=1e-12, atol=1e-12)
            assert all(0 <= row[name] <= 1 for name in ('baseline', *models))
        metrics = summarize(rows)
        assert metrics == json.loads(z.read('summary.json'))['file_metrics']
        log = z.read('run.log').decode()
        assert 'Traceback (most recent call last)' not in log and 'done 48 / 48' in log
        audit = dict(result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            verified_inputs=len(rows), maximum_probe_reconstruction_error=maximum_error,
            metrics=metrics, total_recorded_seconds=sum(r['seconds'] for r in rows),
            segment_counts=sorted(set(segments)), runtime=runtime,
            saturation_ge_0999={name:sum(r[name]>=.999 for r in rows) for name in models},
            gpu_embeddings_independently_recomputed=False,
            baseline_logits_independently_recomputed=False,
            decision='Do not replace the submission model with either probe based on this external check.')
    out = ROOT/'data/experiments/external-music-20260919'
    out.mkdir(parents=True, exist_ok=True)
    target = out/source.name
    if target.exists():
        assert target.read_bytes() == source.read_bytes()
    else:
        shutil.copy2(source, target)
    (out/'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
