"""Package reviewed inputs and the current local inference code for Colab."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'data/research/component-labels-20260920/diagnostic_plan.json').read_text(encoding='utf-8'))
    files = {'script.py':(ROOT/'submit/script.py').read_bytes(),
             'run_component_diagnostic.py':(ROOT/'scripts/run_component_diagnostic.py').read_bytes()}
    tree = ast.parse(files['script.py'].decode('utf-8'))
    config = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id=='CONFIG' for t in n.targets))
    assert config['pipeline']=='separate' and config['file_head']=='direct'
    assert config['music_head']=='none' and config['voice_head']=='none'
    for p in (ROOT/'submit/model').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.yaml','.csv','.txt') and '__pycache__' not in p.parts:
            files['model/'+p.relative_to(ROOT/'submit/model').as_posix()] = p.read_bytes()
    for row in plan['rows']:
        blob = (ROOT/row['clip_path']).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==row['clip_sha256']
        row['clip_path']='inputs/'+row['ID']+'.wav'
        files[row['clip_path']]=blob
    for dataset in ('component-music-20260920','component-remainder-20260920'):
        files['provenance/'+dataset+'.json']=(ROOT/'data/raw'/dataset/'reviewed_manifest.json').read_bytes()
    for p in (ROOT/'data/raw/component-music-20260920/licenses').glob('*.txt'):
        files['provenance/'+p.name]=p.read_bytes()
    with zipfile.ZipFile(Path.home()/'Downloads/component_remaining_review.zip') as z:
        for name in ('ATTRIBUTION.txt','WaveFake_LICENSE.txt'):
            files['provenance/'+name]=z.read(name)
    plan.update(config=config, file_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()})
    files['plan.json']=json.dumps(plan,ensure_ascii=False,indent=2).encode('utf-8')
    target=Path.home()/'Downloads/component_diagnostic_bundle.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name,blob in files.items():
            z.writestr(name,blob)
    bundle_sha=hashlib.sha256(target.read_bytes()).hexdigest()
    cells=[]
    def cell(kind, source):
        item=dict(cell_type=kind,metadata={},source=source.splitlines(True))
        if kind=='code':
            ast.parse(source)
            item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown', '# 성분별 출력 진단: 확인된 19개 음원\nT4 GPU를 선택하고 모두 실행하세요. component_diagnostic_bundle.zip을 업로드합니다.\n가중치 약 5GB를 다운로드할 수 있습니다. 학습·대회 제출은 하지 않습니다. 마지막에 component_results.zip을 내려받습니다.\n로컬에서는 GPU 실행을 하지 않았습니다. Colab 결과로 실제 실행 여부를 확인합니다.\n')
    cell('code', '''from pathlib import Path
import hashlib, io, json, shutil, subprocess, sys, tempfile, zipfile
from google.colab import files
import torch
assert torch.cuda.is_available(), "런타임 → 런타임 유형 변경 → T4 GPU를 선택하세요."
print(torch.cuda.get_device_name(0), torch.__version__, sys.version)
uploaded = files.upload()
assert len(uploaded)==1, "component_diagnostic_bundle.zip 하나만 선택하세요."
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest()=="BUNDLE_SHA", "다른 버전의 ZIP입니다."
WORK = Path(tempfile.mkdtemp(prefix="component_diagnostic_", dir="/content"))
with zipfile.ZipFile(io.BytesIO(blob)) as archive:
    assert archive.testzip() is None
    for name in archive.namelist():
        assert (WORK/name).resolve().is_relative_to(WORK.resolve())
    archive.extractall(WORK)
RESULTS = WORK/"component_results"
RESULTS.mkdir()
plan=json.loads((WORK/"plan.json").read_text())
for name, digest in plan["file_sha256"].items():
    assert hashlib.sha256((WORK/name).read_bytes()).hexdigest()==digest, name
def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4*1024**2),b""):
            h.update(block)
    return h.hexdigest()
print("19개 입력과 실행 코드 무결성 확인 완료", WORK)
'''.replace('BUNDLE_SHA',bundle_sha))
    cell('code', '''packages=["demucs==4.0.1","panns-inference==0.1.1","librosa==0.10.2.post1",
          "transformers==4.57.6","accelerate==1.9.0","einops==0.8.1","soundfile","soxr","scikit-learn"]
installed=subprocess.run([sys.executable,"-m","pip","install",*packages],capture_output=True,text=True)
(RESULTS/"install.log").write_text(installed.stdout+installed.stderr)
print((installed.stdout+installed.stderr)[-4000:])
if installed.returncode:
    files.download(shutil.make_archive("/content/component_results","zip",RESULTS))
    installed.check_returncode()
(RESULTS/"environment.txt").write_text(subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)+"\\n"+sys.version)
print("설치 완료. Colab은 평가 서버와 동일한 환경이 아닙니다.")
''')
    cell('code', '''import urllib.request
from huggingface_hub import hf_hub_download
MODEL=WORK/"model"
expected={relative:digest for digest,relative in (line.split() for line in (MODEL/"SHA256SUMS.txt").read_text().splitlines() if line.strip())}
urls={"htdemucs/955717e8-8726e21a.th":"https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
      "panns/Cnn14_mAP=0.431.pth":"https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"}
try:
    for relative, checksum in expected.items():
        target=MODEL/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_file() and file_sha256(target)==checksum:
            continue
        if target.is_symlink():
            target.unlink()
        previous=next((p for p in Path("/content").glob("*/model/"+relative)
                       if p!=target and p.is_file() and file_sha256(p)==checksum),None)
        if previous is not None:
            shutil.copy2(previous.resolve(strict=True),target)
        elif relative.startswith("df_arena_1b/"):
            cached=Path(hf_hub_download(repo_id="Speech-Arena-2025/DF_Arena_1B_V_1",
                 revision="fb6ce85de12c2c5a509d89114adaf827dd75f49f",filename="pytorch_model.bin"))
            assert file_sha256(cached)==checksum
            shutil.copy2(cached.resolve(strict=True),target)
        else:
            temporary=target.with_suffix(target.suffix+".download")
            with urllib.request.urlopen(urls[relative],timeout=60) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response,stream,4*1024**2)
            assert file_sha256(temporary)==checksum, relative
            temporary.replace(target)
        assert target.is_file() and file_sha256(target)==checksum, relative
        print("가중치 검사 통과", relative)
    (RESULTS/"download_checks.json").write_text(json.dumps(expected,indent=2))
except Exception:
    import traceback
    (RESULTS/"download_error.txt").write_text(traceback.format_exc())
    files.download(shutil.make_archive("/content/component_results","zip",RESULTS))
    raise
''')
    cell('code', '''try:
    with (RESULTS/"run.log").open("w",encoding="utf-8") as log:
        process=subprocess.Popen([sys.executable,"-u","run_component_diagnostic.py","--work",str(WORK)],
            cwd=WORK,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
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
    assert code==0, "진단 중단: 내려받은 component_results.zip과 오류를 보내주세요."
    report=json.loads((RESULTS/"summary.json").read_text())
    assert report["completed"] and report["predictions"]==19
    print("완료: 19개 처리. 이 결과는 대회 점수가 아닙니다.")
finally:
    files.download(shutil.make_archive("/content/component_results","zip",RESULTS))
''')
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                    accelerator='GPU',colab=dict(name='colab_component_diagnostic.ipynb')),
            nbformat=4,nbformat_minor=5)
    notebook=ROOT/'notebooks/colab_component_diagnostic.ipynb'
    notebook.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    shutil.copy2(notebook,Path.home()/'Downloads'/notebook.name)
    print(target, target.stat().st_size, bundle_sha)
    print('NOTEBOOK',notebook)


if __name__=='__main__':
    main()
