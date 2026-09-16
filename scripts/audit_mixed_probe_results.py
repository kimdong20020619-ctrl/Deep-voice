"""Verify returned features, train-only scaling, probabilities, and development metrics."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np
from run_mixed_probe import validate_manifest, family_metrics
from run_file_probe_experiment import evaluate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from train.fit_probe import predict


def main():
    source = Path.home()/'Downloads/mixed_probe_results.zip'
    with zipfile.ZipFile(source) as z, zipfile.ZipFile(ROOT/'mixed_probe_bundle.zip') as bundle:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        manifest = json.loads(z.read('manifest.json'))
        assert manifest == json.loads(bundle.read('probe/manifest.json'))
        validate_manifest(manifest)
        for name, original in [('script.py','script.py'),('run_mixed_probe.py','run_mixed_probe.py'),
            ('run_file_probe_experiment.py','run_file_probe_experiment.py'),('fit_probe.py','src/train/fit_probe.py'),
            ('SHA256SUMS.txt','model/SHA256SUMS.txt'),('previous_music_probe.npz','previous_music_probe.npz')]:
            assert z.read(name) == bundle.read(original), name
        identity = hashlib.sha256(b''.join(bundle.read(name) for name in
            ('script.py','run_mixed_probe.py','src/train/fit_probe.py','probe/manifest.json','model/SHA256SUMS.txt'))).hexdigest()
        assert identity == json.loads(z.read('extraction_identity.json'))['code_manifest_model_hash']
        rows = manifest['rows']
        train = [r for r in rows if r['split']=='train']
        dev = [r for r in rows if r['split']=='development']
        assert len(rows)==348 and len(train)==268 and len(dev)==80
        features, baseline = {}, {}
        assert {n for n in z.namelist() if n.startswith('features/') and n.endswith('.npz')} == {'features/'+r['ID']+'.npz' for r in rows}
        for row in rows:
            with np.load(io.BytesIO(z.read('features/'+row['ID']+'.npz')),allow_pickle=False) as data:
                assert str(data['clip_sha256']) == row['clip_sha256']
                embedding = data['embeddings']
                assert embedding.ndim==2 and embedding.shape[1]==1280 and len(embedding)>0
                assert np.isfinite(embedding).all()
                features[row['ID']] = embedding
                baseline[row['ID']] = float(data['baseline'])
                assert np.isfinite(baseline[row['ID']]) and 0<=baseline[row['ID']]<=1
        models = {}
        for key, name in [('mixed_probe','experimental_mixed_probe.npz'),('previous_music_probe','previous_music_probe.npz')]:
            with np.load(io.BytesIO(z.read(name)),allow_pickle=False) as data:
                models[key] = {k:data[k].copy() for k in ('w','b','mean','scale')}
            assert all(np.isfinite(v).all() for v in models[key].values())
            assert np.all(models[key]['scale']>0)
        X = np.concatenate([features[r['ID']] for r in train])
        expected_scale = X.std(axis=0)
        expected_scale[expected_scale<1e-8] = 1
        np.testing.assert_array_equal(models['mixed_probe']['mean'],X.mean(axis=0))
        np.testing.assert_array_equal(models['mixed_probe']['scale'],expected_scale)
        scores = {'baseline':baseline}
        for name, model in models.items():
            scores[name] = {r['ID']:float(predict(model,features[r['ID']]).max()) for r in dev}
        predictions = list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
        assert len(predictions)==80 and {r['ID'] for r in predictions}=={r['ID'] for r in dev}
        indexed = {r['ID']:r for r in dev}
        maximum_error = 0.
        for row in predictions:
            assert row['family']==indexed[row['ID']]['family']
            assert int(row['file_fake'])==indexed[row['ID']]['file_fake']
            for name in scores:
                expected = scores[name][row['ID']]
                maximum_error = max(maximum_error,abs(float(row[name])-expected))
                np.testing.assert_allclose(float(row[name]),expected,atol=1e-12,rtol=1e-12)
        metrics = {name:family_metrics(dev,s) for name,s in scores.items()}
        summary = json.loads(z.read('summary.json'))
        for name in metrics:
            assert metrics[name]==summary[name]
        selection = json.loads(z.read('selection.json'))
        selected = min(selection['candidates'],key=lambda c:(c['metrics']['macro_family_eer'],c['metrics']['all']['file_eer'],c['C']))
        assert selected['C']==selection['selected_C'] and selected['metrics']==metrics['mixed_probe']
        sources = {s['ID']:s for s in manifest['sources']}
        subgroups = {}
        for family in ('mix_voice_-6dB','mix_voice_+6dB'):
            for fake_kind in ('music','speech'):
                subset = [r for r in dev if r['family']==family and (not r['file_fake'] or
                    any(sources[s]['kind']==fake_kind and sources[s]['file_fake'] for s in r['source_ids']))]
                subgroups[family+'/'+fake_kind+'_fake'] = {
                    name:evaluate([r['file_fake'] for r in subset],[s[r['ID']] for r in subset]) for name,s in scores.items()}
        assert 'Traceback (most recent call last)' not in z.read('run.log').decode()
        audit = dict(result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(z.read('experimental_mixed_probe.npz')).hexdigest(),
            feature_files=len(rows), train_segments=len(X), development_files=len(dev),
            maximum_prediction_reconstruction_error=maximum_error, train_only_scaling_verified=True,
            selected_C=selection['selected_C'], candidate_selection_rule_verified=True,
            unselected_candidate_weights_independently_refit=False,
            metrics=metrics, mixed_component_diagnostics=subgroups)
    out = ROOT/'data/experiments/mixed-probe-20260916'
    out.mkdir(parents=True,exist_ok=True)
    target = out/source.name
    if target.exists():
        assert target.read_bytes()==source.read_bytes()
    else:
        shutil.copy2(source,target)
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
