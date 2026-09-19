"""Build a pinned SONICS screening bundle and Colab notebook."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RESEARCH=ROOT/'data/research/sonics-20260919'


def cell(kind, text):
    item=dict(cell_type=kind,metadata={},source=text.splitlines(True))
    if kind=='code': item.update(execution_count=None,outputs=[])
    return item


def main():
    files={}
    rows=[]
    for tag, bundle_name, result_path in (
        ('dev','separation_bundle.zip','data/experiments/separation-20260916/separation_results.zip'),
        ('ext','external_music_bundle.zip','data/experiments/external-music-20260919/external_music_results.zip')):
        with zipfile.ZipFile(ROOT/bundle_name) as z,zipfile.ZipFile(ROOT/result_path) as result:
            manifest=json.loads(z.read('probe/manifest.json'))
            assert manifest==json.loads(result.read('manifest.json'))
            predictions={r['ID']:r for r in csv.DictReader(io.StringIO(result.read('predictions.csv').decode()))}
            for row in manifest['rows']:
                identity=tag+'_'+row['ID']
                blob=z.read('probe/test/'+row['ID']+'.wav')
                assert hashlib.sha256(blob).hexdigest()==row['clip_sha256']
                path='audio/'+identity+'.wav'
                files[path]=blob
                cohort='external' if tag=='ext' else ('development_music' if row['music_comparison_eligible'] else 'development_controls')
                rows.append(dict(ID=identity,path=path,cohort=cohort,family=row.get('family','broadcast'),file_fake=row['file_fake'],
                    clip_sha256=row['clip_sha256'],reference_baseline=float(predictions[row['ID']]['baseline' if tag=='ext' else 'original_baseline'])))
            files['provenance/'+tag+'_manifest.json']=json.dumps(manifest,indent=2).encode()
            for name in z.namelist():
                if name.startswith('probe/') and name.endswith(('.txt','.json')) and name!='probe/manifest.json':
                    files['provenance/'+tag+'/'+name.removeprefix('probe/')]=z.read(name)
            files['provenance/'+tag+'_result_sha256.txt']=hashlib.sha256((ROOT/result_path).read_bytes()).hexdigest().encode()
    for path in (RESEARCH/'upstream/sonics').rglob('*.py'):
        files[path.relative_to(RESEARCH/'upstream').as_posix()]=path.read_bytes()
    for name in ('LICENSE','README.md','test.py','requirements.txt'):
        files['provenance/sonics/'+name]=(RESEARCH/'upstream'/name).read_bytes()
    files['provenance/sonics/receipts.json']=(RESEARCH/'receipts.json').read_bytes()
    models={}
    for variant in ('alpha-5s','alpha-120s'):
        meta=json.loads((RESEARCH/variant/'metadata.json').read_text())
        weight=next(x for x in meta['siblings'] if x['rfilename']=='pytorch_model.bin')
        assert meta['cardData']['license']=='mit'
        config=(RESEARCH/variant/'config.json').read_bytes()
        files['weights/'+variant+'/config.json']=config
        files['provenance/sonics/'+variant+'_metadata.json']=(RESEARCH/variant/'metadata.json').read_bytes()
        files['provenance/sonics/'+variant+'_README.md']=(RESEARCH/variant/'README.md').read_bytes()
        models[variant]=dict(repo=meta['id'],revision=meta['sha'],weights_sha256=weight['lfs']['sha256'],
            weights_size=weight['size'],config_sha256=hashlib.sha256(config).hexdigest())
    for name in ('run_sonics_check.py','run_file_probe_experiment.py'):
        files[name]=(ROOT/'scripts'/name).read_bytes()
    plan=dict(rows=rows,models=models,source_revision='9156ffad151f797c71556923c4a02fa01fa8fc91',
        primary='alpha-5s, 5s windows, mean probabilities; max is diagnostic only',
        secondary='alpha-120s only on external 60s audio, right-zero-pad to 120s per official deterministic dataset loader',
        fit=False,expected_predictions=352,
        limitations='Previously inspected development data. SONICS training includes Suno; external corpus is not proof of unseen generator or recording.',
        file_sha256={k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
    assert len(rows)==128 and len({r['ID'] for r in rows})==128
    files['plan.json']=json.dumps(plan,indent=2).encode()
    output=ROOT/'sonics_check_bundle.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        for name,blob in files.items(): z.writestr(name,blob)
    sha=hashlib.sha256(output.read_bytes()).hexdigest()
    cells=[cell('markdown','# SONICS 음악 탐지 모델 검증\nT4 GPU에서 모두 실행하고 sonics_check_bundle.zip 하나를 선택하세요.\n새 패키지 timm 설치는 사용자 승인에 따라 Colab에만 수행합니다. 학습하거나 DACON에 제출하지 않습니다.\n5초 모델이 주 비교 대상이며 120초 모델은 길이 불일치를 확인하는 보조 실험입니다. 결과: sonics_results.zip\n')]
    cells.append(cell('code','''from google.colab import files
from pathlib import Path
import torch, sys, json, hashlib, zipfile, io, tempfile, shutil, subprocess, os
assert torch.cuda.is_available(), "런타임 유형을 T4 GPU로 바꾸세요."
print(torch.cuda.get_device_name(0), torch.__version__, sys.version)
uploaded=files.upload()
assert len(uploaded)==1, "sonics_check_bundle.zip 하나만 선택하세요."
blob=next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest()=="BUNDLE_SHA", "ZIP 버전이 다릅니다."
WORK=Path(tempfile.mkdtemp(prefix="sonics_check_",dir="/content"))
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for name in z.namelist():
        assert (WORK/name).resolve().is_relative_to(WORK.resolve())
    z.extractall(WORK)
plan=json.loads((WORK/"plan.json").read_text())
for name,sha in plan["file_sha256"].items():
    assert hashlib.sha256((WORK/name).read_bytes()).hexdigest()==sha, name
RESULTS=WORK/"sonics_results"
RESULTS.mkdir()
print("128개 입력과 실행 코드 무결성 확인 완료")
'''.replace('BUNDLE_SHA',sha)))
    cells.append(cell('code','''# torch/torchvision을 임의로 업그레이드하지 않는다.
installed=subprocess.run([sys.executable,"-m","pip","install","--no-deps","timm==1.0.15"],capture_output=True,text=True)
(RESULTS/"install.log").write_text(installed.stdout+installed.stderr)
print((installed.stdout+installed.stderr)[-4000:])
installed.check_returncode()
check=subprocess.run([sys.executable,"-c","import timm, torch, torchvision, torchaudio, librosa, soundfile, huggingface_hub; from sonics import HFAudioClassifier; print('의존성 검사 통과', timm.__version__)"],cwd=WORK,capture_output=True,text=True)
(RESULTS/"preflight.log").write_text(check.stdout+check.stderr)
print(check.stdout+check.stderr)
check.check_returncode()
assert shutil.which("ffmpeg"), "ffmpeg가 없습니다."
(RESULTS/"environment.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)+"\\n"+sys.version)
'''))
    cells.append(cell('code','''from huggingface_hub import hf_hub_download
for variant,entry in plan["models"].items():
    cached=Path(hf_hub_download(repo_id=entry["repo"],filename="pytorch_model.bin",revision=entry["revision"]))
    assert cached.stat().st_size==entry["weights_size"]
    assert hashlib.sha256(cached.read_bytes()).hexdigest()==entry["weights_sha256"], variant
    shutil.copy2(cached,WORK/"weights"/variant/"pytorch_model.bin")
    print(variant,"가중치 해시 검사 통과")
'''))
    cells.append(cell('code','''# 추론은 로컬 가중치를 사용하며 HF 네트워크 다운로드를 금지한다.
env=os.environ.copy()
env["HF_HUB_OFFLINE"]="1"
env["TRANSFORMERS_OFFLINE"]="1"
try:
    with (RESULTS/"run.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,"-u","run_sonics_check.py","--work",str(WORK)],cwd=WORK,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end="")
                log.write(line)
                log.flush()
            code=process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    assert code==0, "검증이 중단됐습니다. 결과 ZIP과 오류 내용을 보내주세요."
finally:
    archive=shutil.make_archive("/content/sonics_results","zip",RESULTS)
    files.download(archive)
'''))
    nb=dict(cells=cells,metadata=dict(accelerator='GPU',kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    for c in cells:
        if c['cell_type']=='code': ast.parse(''.join(c['source']))
    (ROOT/'notebooks/colab_sonics_check.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('READY',len(rows),output.stat().st_size,sha)


if __name__=='__main__':
    main()
