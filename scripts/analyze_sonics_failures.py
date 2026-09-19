"""Descriptive error analysis only: no fitted thresholds, fusion or model selection."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from run_file_probe_experiment import evaluate

ROOT=Path(__file__).resolve().parents[1]


def measured(labels,scores):
    if len(set(labels))<2:
        return dict(n=len(labels),file_eer=None,file_auc=None,reason='Only one class; EER/AUC undefined')
    return evaluate(labels,scores)


def main():
    source=ROOT/'data/experiments/sonics-20260919/sonics_results.zip'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='53e6db5540cf29d223a0698dbdfca88338704356ee4f37e21e788c7685d25da0'
    with zipfile.ZipFile(source) as z:
        plan=json.loads(z.read('plan.json'))
        dev=json.loads(z.read('provenance/dev_manifest.json'))
        ext=json.loads(z.read('provenance/ext_manifest.json'))
        sources={r['ID']:r for r in dev['sources']}
        metadata={'dev_'+r['ID']:r for r in dev['rows']}
        metadata.update({'ext_'+r['ID']:r for r in ext['rows']})
        rows=list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))
        for r in rows:
            r['file_fake']=int(r['file_fake'])
            for k in ('mean_prob','max_prob','reference_baseline'): r[k]=float(r[k])
        original=[r for r in rows if r['model']=='alpha-5s' and r['view']=='original']
        compressed={r['ID']:r for r in rows if r['model']=='alpha-5s' and r['view']=='mp3_64k'}
        report={}
        for cohort in sorted({r['cohort'] for r in original}):
            for family in sorted({r['family'] for r in original if r['cohort']==cohort}):
                group=[r for r in original if r['cohort']==cohort and r['family']==family]
                labels=[r['file_fake'] for r in group]
                entry={head:measured(labels,[r[head] for r in group]) for head in ('mean_prob','reference_baseline')}
                entry['mp3']=measured(labels,[compressed[r['ID']]['mean_prob'] for r in group])
                entry['by_class']={}
                for label in (0,1):
                    selected=[r for r in group if r['file_fake']==label]
                    if not selected:
                        continue
                    entry['by_class'][str(label)]=dict(n=len(selected),mean_score=float(np.mean([r['mean_prob'] for r in selected])),
                        mean_mp3_change=float(np.mean([compressed[r['ID']]['mean_prob']-r['mean_prob'] for r in selected])))
                report[cohort+'/'+family]=entry
        music=[r for r in original if r['family']=='music']
        generator_counts={}
        for r in music:
            for identity in metadata[r['ID']]['source_ids']:
                s=sources[identity]
                group=s.get('generator',s.get('group','unknown'))
                generator_counts.setdefault(str(group),[]).append(r['mean_prob'])
        external=[r for r in original if r['cohort']=='external']
        paired={}
        for r in external:
            paired.setdefault(metadata[r['ID']]['pair_id'],{})[r['file_fake']]=r
        pair_stats={head:dict(wins=sum(p[1][head]>p[0][head] for p in paired.values()),ties=sum(p[1][head]==p[0][head] for p in paired.values()),n=len(paired)) for head in ('mean_prob','reference_baseline')}
        segments=[]
        for r in external:
            with np.load(io.BytesIO(z.read('segments/'+r['ID']+'_alpha-5s_original.npz')),allow_pickle=False) as a:
                p=1/(1+np.exp(-a['logits']))
                assert len(p)==12
                segments.append(p)
        segments=np.array(segments)
        labels=[r['file_fake'] for r in external]
        temporal={'first_5s':evaluate(labels,segments[:,0]),'first_10s_mean':evaluate(labels,segments[:,:2].mean(1)),
            'full_60s_mean':evaluate(labels,segments.mean(1)),
            'per_5s_position':[evaluate(labels,segments[:,i]) for i in range(12)]}
        # Resample complete real/fake pairs, not correlated individual rows.
        pairs=list(paired.values())
        rng=np.random.default_rng(20260919)
        deltas=[]
        for _ in range(1000):
            sample=[pairs[i][label] for i in rng.integers(0,len(pairs),len(pairs)) for label in (0,1)]
            y=[r['file_fake'] for r in sample]
            a=evaluate(y,[r['mean_prob'] for r in sample])
            b=evaluate(y,[r['reference_baseline'] for r in sample])
            deltas.append([a['file_eer']-b['file_eer'],a['file_auc']-b['file_auc']])
        interval=np.quantile(deltas,[.025,.975],axis=0)
        output=dict(family_metrics=report,pure_music_source_scores={g:dict(n=len(p),mean_score=float(np.mean(p))) for g,p in generator_counts.items()},
            external_pair_ranking=pair_stats,external_temporal_diagnostics=temporal,
            external_paired_bootstrap=dict(replicates=1000,seed=20260919,eer_difference_95_percentile=interval[:,0].tolist(),auc_difference_95_percentile=interval[:,1].tolist(),
                note='Exploratory percentile intervals on previously inspected 24 pairs, not a confirmatory significance test.'),
            data_sources=sorted({s.get('generator','unknown') for s in dev['sources'] if s.get('file_fake')==1 and s.get('kind')=='music'}),
            fit=False,threshold_selection=False,submission_changed=False)
    target=ROOT/'data/experiments/sonics-20260919/failure_analysis.json'
    target.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
