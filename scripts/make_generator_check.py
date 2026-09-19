"""Package five-generator, two-duration, matched-codec screening."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files={}
    with zipfile.ZipFile(ROOT/'sonics_check_bundle.zip') as sonics:
        old_plan=json.loads(sonics.read('plan.json'))
        for name in sonics.namelist():
            if name.startswith(('sonics/','provenance/sonics/','weights/alpha-5s/')):
                files[name]=sonics.read(name)
    with zipfile.ZipFile(ROOT/'file_probe_bundle.zip') as base:
        original=json.loads(base.read('probe/manifest.json'))
        rows=[]
        for r in original['rows']:
            blob=base.read('probe/test/'+r['ID']+'.wav')
            assert hashlib.sha256(blob).hexdigest()==r['clip_sha256']
            path='audio/'+r['ID']+'.wav'
            files[path]=blob
            rows.append(dict(ID=r['ID'],path=path,file_fake=r['file_fake'],generator=r.get('generator','human'),
                historical_split=r['split'],clip_sha256=r['clip_sha256'],source_sha256=r['source_sha256'],source_group=r['group']))
        for name in base.namelist():
            if name.startswith('probe/') and name.endswith('.txt'):
                files['provenance/data/'+name.removeprefix('probe/')]=base.read(name)
        files['provenance/data/manifest.json']=base.read('probe/manifest.json')
    with zipfile.ZipFile(ROOT/'separation_bundle.zip') as base:
        for name in base.namelist():
            if name.startswith('model/df_arena_1b/') or name=='script.py':
                files[name]=base.read(name)
    for name in ('run_generator_check.py','run_sonics_check.py','run_file_probe_experiment.py'):
        files[name]=(ROOT/'scripts'/name).read_bytes()
    df_sha='780bc14fd4c15e65d58efdef728427cf03cd29cd60be528e97badf8c89087988'
    plan=dict(rows=rows,models={'alpha-5s':old_plan['models']['alpha-5s']},
        df_model=dict(repo='Speech-Arena-2025/DF_Arena_1B_V_1',revision='fb6ce85de12c2c5a509d89114adaf827dd75f49f'),
        weight_sha256=dict(sonics=old_plan['models']['alpha-5s']['weights_sha256'],df_arena=df_sha),
        expected_predictions=336,fit=False,
        interpretation='All are previously used samples, including historical holdout. Generator comparisons share 41 REAL controls. No new independent holdout.',
        file_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    assert len(rows)==84 and len({r['clip_sha256'] for r in rows})==84
    files['plan.json']=json.dumps(plan,indent=2).encode()
    output=ROOT/'generator_check_bundle.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,blob in files.items(): z.writestr(name,blob)
    sha=hashlib.sha256(output.read_bytes()).hexdigest()
    nb=json.loads((ROOT/'notebooks/colab_sonics_check.ipynb').read_text(encoding='utf-8'))
    nb['cells'][0]['source']=['# 생성기·길이·MP3 조건을 맞춘 비교\n','T4 GPU에서 모두 실행하고 generator_check_bundle.zip을 선택하세요.\n',
        '기존 음원 84개를 두 길이·두 코덱 조건으로 처리하고 SONICS와 DF-Arena를 비교합니다. 과거 holdout도 이미 사용한 개발 자료로 취급합니다. 학습·대회 제출은 하지 않습니다.\n',
        'DF-Arena 가중치 다운로드에 약 4.6GB가 필요할 수 있습니다. 결과: generator_results.zip\n']
    source=''.join(nb['cells'][1]['source'])
    old=source.split('hexdigest()=="')[1].split('"')[0]
    source=source.replace(old,sha).replace('sonics_check_bundle','generator_check_bundle').replace('sonics_check_','generator_check_').replace('sonics_results','generator_results').replace('128개','84개')
    nb['cells'][1]['source']=source.splitlines(True)
    install=''.join(nb['cells'][2]['source'])
    install=install.replace('check=subprocess.run', '''extra=subprocess.run([sys.executable,"-m","pip","install","demucs==4.0.1"],capture_output=True,text=True)
with (RESULTS/"install.log").open("a") as log:
    log.write(extra.stdout+extra.stderr)
print((extra.stdout+extra.stderr)[-2000:])
extra.check_returncode()
check=subprocess.run''')
    install=install.replace('from sonics import HFAudioClassifier;','from sonics import HFAudioClassifier; import script;')
    nb['cells'][2]['source']=install.splitlines(True)
    download=''.join(nb['cells'][3]['source'])+'''
entry=plan["df_model"]
cached=Path(hf_hub_download(repo_id=entry["repo"],filename="pytorch_model.bin",revision=entry["revision"]))
h=hashlib.sha256()
with cached.open("rb") as stream:
    for block in iter(lambda:stream.read(4*1024**2),b""):
        h.update(block)
assert h.hexdigest()==plan["weight_sha256"]["df_arena"]
target=WORK/"model/df_arena_1b/pytorch_model.bin"
try:
    os.link(cached,target)
except OSError:
    shutil.copy2(cached,target)
print("DF-Arena 가중치 해시 검사 통과")
'''
    nb['cells'][3]['source']=download.splitlines(True)
    runner=''.join(nb['cells'][4]['source']).replace('run_sonics_check.py','run_generator_check.py').replace('sonics_results','generator_results')
    nb['cells'][4]['source']=runner.splitlines(True)
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    (ROOT/'notebooks/colab_generator_check.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('READY',output.stat().st_size,sha,dict(Counter(r['generator'] for r in rows)))


if __name__=='__main__':
    main()
