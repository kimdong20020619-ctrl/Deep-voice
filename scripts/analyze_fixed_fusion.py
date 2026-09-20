"""One fixed equal-probability fusion; descriptive development analysis only."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from run_file_probe_experiment import evaluate

ROOT=Path(__file__).resolve().parents[1]


def compare(rows, sonics='sonics_mean', baseline='df_arena'):
    y=np.array([int(r['file_fake']) for r in rows])
    a=np.array([float(r[sonics]) for r in rows])
    b=np.array([float(r[baseline]) for r in rows])
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert ((0<=a)&(a<=1)&(0<=b)&(b<=1)).all()
    fused=(a+b)/2
    metrics={k:evaluate(y,s) for k,s in [('sonics',a),('df_arena',b),('equal_mean',fused)]}
    # Rank pairs, avoiding any fitted classification threshold. These pairs share samples.
    ma=a[y==1,None]-a[y==0]
    mb=b[y==1,None]-b[y==0]
    mf=fused[y==1,None]-fused[y==0]
    ranks=dict(real_fake_pairs=int(ma.size),both_strictly_correct=int(((ma>0)&(mb>0)).sum()),
        sonics_only_strictly_correct=int(((ma>0)&(mb<=0)).sum()),
        df_only_strictly_correct=int(((mb>0)&(ma<=0)).sum()),
        neither_strictly_correct=int(((ma<=0)&(mb<=0)).sum()),
        fusion_strictly_correct=int((mf>0).sum()),fusion_ties=int((mf==0).sum()),
        note='Repeated samples across rank pairs; counts are not independent observations.')
    return dict(metrics=metrics,pair_ranking=ranks,
        delta_eer_vs_df=metrics['equal_mean']['file_eer']-metrics['df_arena']['file_eer'],
        delta_auc_vs_df=metrics['equal_mean']['file_auc']-metrics['df_arena']['file_auc'])


def read_result(path,sha):
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    with zipfile.ZipFile(path) as z:
        return list(csv.DictReader(io.StringIO(z.read('predictions.csv').decode())))


def main():
    rows=read_result(ROOT/'data/experiments/generator-20260919/generator_results.zip',
        '3bfaf6cbc1e3cb8ebd9a351a10a639d24c51d66d8744da677844add07a69403f')
    report={}
    for seconds in ('5','8'):
        for view in ('original','mp3_64k'):
            condition=[r for r in rows if r['seconds_input']==seconds and r['view']==view]
            assert len(condition)==84
            for generator in sorted({r['generator'] for r in condition if r['file_fake']=='1'}):
                group=[r for r in condition if r['file_fake']=='0' or r['generator']==generator]
                report[f'{seconds}s/{view}/{generator}']=compare(group)
    previous=read_result(ROOT/'data/experiments/sonics-20260919/sonics_results.zip',
        '53e6db5540cf29d223a0698dbdfca88338704356ee4f37e21e788c7685d25da0')
    controls={}
    for cohort in ('external','development_music','development_controls'):
        group=[r for r in previous if r['model']=='alpha-5s' and r['view']=='original' and r['cohort']==cohort]
        controls[cohort]=compare(group,'mean_prob','reference_baseline')
    decisions=dict(conditions=20,
        strict_both_improve=sum(r['delta_eer_vs_df']<0 and r['delta_auc_vs_df']>0 for r in report.values()),
        eer_worse=sum(r['delta_eer_vs_df']>0 for r in report.values()),
        auc_worse=sum(r['delta_auc_vs_df']<0 for r in report.values()))
    output=dict(rule='0.5 * SONICS probability + 0.5 * DF-Arena probability',fit=False,weight_search=False,
        generator_comparison=report,previous_original_controls=controls,summary=decisions,
        caveat='All previously inspected development data; no independent holdout or competition Score. Previous controls use archived DF predictions from a different run.',
        decision='Diagnostic only; no submission change or automatic adoption.')
    out=ROOT/'data/experiments/fixed-fusion-20260920'
    out.mkdir(parents=True,exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(decisions))
    for name,result in {**report,**controls}.items():
        print(name,[(k,round(v['file_eer'],4),round(v['file_auc'],4)) for k,v in result['metrics'].items()])
    # Analytic fixtures ensure averaging and pair counts cannot silently swap labels.
    fixture=[dict(file_fake=0,sonics_mean=.2,df_arena=.4),dict(file_fake=1,sonics_mean=.8,df_arena=.6)]
    test=compare(fixture)
    assert test['metrics']['equal_mean']['file_eer']==0 and test['metrics']['equal_mean']['file_auc']==1
    assert test['pair_ranking']['both_strictly_correct']==1
    reverse=compare([dict(file_fake=0,sonics_mean=1.,df_arena=0.),dict(file_fake=1,sonics_mean=0.,df_arena=1.)])
    assert reverse['metrics']['equal_mean']['file_eer']==.5 and reverse['pair_ranking']['fusion_ties']==1
    print('Analytic fixtures passed')


if __name__=='__main__':
    main()
