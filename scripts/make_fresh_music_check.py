"""Build a FILE-only screening bundle; preserve unknown component labels."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/research/fresh-music-check-20260923'


def main():
    files={}
    with zipfile.ZipFile(ROOT/'generator_check_bundle.zip') as old:
        old_plan=json.loads(old.read('plan.json'))
        for name in old.namelist():
            if name.startswith(('sonics/','provenance/sonics/','weights/alpha-5s/')):
                assert not name.endswith('.bin')
                files[name]=old.read(name)
    rows=[]
    for dataset in ('citizen-dj-20260923','fresh-fake-music-20260923'):
        folder=ROOT/'data/raw'/dataset
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        files['provenance/'+dataset+'/manifest.json']=(folder/'manifest.json').read_bytes()
        for source in manifest['rows']:
            blob=(ROOT/source['clip_path']).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==source['clip_sha256']
            path='audio/'+source['ID']+'.wav'
            files[path]=blob
            rows.append(dict(ID=source['ID'],path=path,file_fake=source['file_fake'],
                generator=source.get('generator','human'),historical_split='new_source_screening',
                clip_sha256=source['clip_sha256'],source_sha256=source['source_sha256'],
                source_group=source.get('caption_group',source.get('artist_group')),
                voice_presence=None,music_presence=None,voice_fake=None,music_fake=None))
            if dataset.startswith('citizen'):
                name=source['source_id']+'.md'
                files['provenance/'+dataset+'/'+name]=(folder/'metadata'/name).read_bytes()
        for name in ('ATTRIBUTION.txt','rights.md','record.json','selection.json','commit.json'):
            if (folder/name).exists():files['provenance/'+dataset+'/'+name]=(folder/name).read_bytes()
    for p in (ROOT/'submit/model/df_arena_1b').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.yaml','.txt','.md') and '__pycache__' not in p.parts:
            files['model/df_arena_1b/'+p.relative_to(ROOT/'submit/model/df_arena_1b').as_posix()]=p.read_bytes()
    files['script.py']=(ROOT/'submit/script.py').read_bytes()
    for name in ('run_fresh_music_check.py','run_sonics_check.py','run_file_probe_experiment.py'):
        files[name]=(ROOT/'scripts'/name).read_bytes()
    tree=ast.parse(files['script.py'].decode('utf-8'))
    config=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    assert config['file_head']=='direct' and config['segment_agg']=='max'
    assert len(rows)==29 and len({r['ID'] for r in rows})==29
    assert len({r['clip_sha256'] for r in rows})==29
    assert sum(r['file_fake']==0 for r in rows)==9
    plan=dict(rows=rows,models=old_plan['models'],df_model=old_plan['df_model'],
        weight_sha256=old_plan['weight_sha256'],df_config=config,expected_predictions=58,fit=False,
        evaluation=dict(target='FILE_FAKE only',duration_seconds=8,views=['original','mp3_64k'],
            tuning_allowed=False,component_metrics_allowed=False,competition_score_allowed=False,
            screening_rule='SONICS fixed probability mean must strictly improve both FILE EER and AUC in aggregate and in every generator/view comparison; ties do not pass. Passing is not deployment approval.',
            limitation='9 real sources shared across comparisons; class-source confounding, artist aliases and pretraining overlap unverified. Component review pending.'),
        file_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    blob=json.dumps(plan,ensure_ascii=False,indent=2).encode('utf-8')
    files['plan.json']=blob
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'plan.json').write_bytes(blob)
    downloads=Path.home()/'Downloads'
    bundle=downloads/'fresh_music_check_bundle.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)
    digest=hashlib.sha256(bundle.read_bytes()).hexdigest()
    nb=json.loads((ROOT/'notebooks/colab_generator_check.ipynb').read_text(encoding='utf-8'))
    nb['cells'][0]['source']=['# 새 외부 음원의 파일 진위 비교\n',
        'T4 GPU에서 모두 실행하고 fresh_music_check_bundle.zip을 선택하세요.\n',
        '실제 음악 후보 9개와 생성 음악 20개의 8초 구간을 원본/MP3 조건에서 비교합니다.\n',
        '청취 미완료이므로 음악/음성 성분 지표나 대회 총점은 계산하지 않습니다. 학습·대회 제출 없음.\n',
        'DF-Arena와 SONICS 가중치 다운로드가 필요할 수 있습니다. 결과: fresh_music_results.zip\n']
    source=''.join(nb['cells'][1]['source'])
    previous=source.split('hexdigest()=="')[1].split('"')[0]
    source=source.replace(previous,digest).replace('generator_check','fresh_music_check').replace('generator_results','fresh_music_results').replace('84개','29개')
    source=source.replace('    z.extractall(WORK)','    assert z.testzip() is None\n    z.extractall(WORK)')
    nb['cells'][1]['source']=source.splitlines(True)
    install=''.join(nb['cells'][2]['source'])
    install=install.replace('import script;','import script, transformers, sklearn;')
    nb['cells'][2]['source']=('try:\n'+textwrap.indent(install,'    ')+
        '\nexcept Exception:\n    import traceback\n    (RESULTS/"setup_error.txt").write_text(traceback.format_exc())\n    files.download(shutil.make_archive("/content/fresh_music_results","zip",RESULTS))\n    raise\n').splitlines(True)
    download='''from huggingface_hub import hf_hub_download

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):
            h.update(block)
    return h.hexdigest()

try:
    entries=[(entry,WORK/'weights'/variant/'pytorch_model.bin',entry['weights_sha256'])
             for variant,entry in plan['models'].items()]
    entries.append((plan['df_model'],WORK/'model/df_arena_1b/pytorch_model.bin',plan['weight_sha256']['df_arena']))
    for entry,target,expected in entries:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and digest_file(target)==expected:
            continue
        if target.is_symlink():
            target.unlink()
        cached=Path(hf_hub_download(repo_id=entry['repo'],filename='pytorch_model.bin',revision=entry['revision']))
        assert digest_file(cached)==expected
        shutil.copy2(cached.resolve(strict=True),target)
        assert digest_file(target)==expected
        print('가중치 확인 완료',entry['repo'])
except Exception:
    import traceback
    (RESULTS/'download_error.txt').write_text(traceback.format_exc())
    files.download(shutil.make_archive('/content/fresh_music_results','zip',RESULTS))
    raise
'''
    nb['cells'][3]['source']=download.splitlines(True)
    runner=''.join(nb['cells'][4]['source']).replace('run_generator_check.py','run_fresh_music_check.py').replace('generator_results','fresh_music_results')
    runner=runner.replace('finally:\n    archive=', '''    summary=json.loads((RESULTS/'summary.json').read_text())
    assert summary['completed'] and summary['predictions']==58
    print('완료: 파일 진위 비교. 성분별 성능/대회 총점이 아닙니다.')
finally:
    archive=''')
    nb['cells'][4]['source']=runner.splitlines(True)
    nb['metadata'].setdefault('colab',{})['name']='colab_fresh_music_check.ipynb'
    for i,c in enumerate(nb['cells']):
        c['id']=f'fresh-music-{i}'
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
            c['execution_count']=None;c['outputs']=[]
    notebook=ROOT/'notebooks/colab_fresh_music_check.ipynb'
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    shutil.copy2(notebook,downloads/notebook.name)
    print('READY',bundle,bundle.stat().st_size,digest)
    print('NOTEBOOK',downloads/notebook.name)


if __name__=='__main__':
    main()
