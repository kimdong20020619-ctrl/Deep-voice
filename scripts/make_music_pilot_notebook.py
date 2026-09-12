"""Build a self-contained Colab FILE diagnostic pilot and small upload bundle."""

import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/raw/music-pilot-20260912"
OUT = REPO / "music_pilot_bundle.zip"
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
assert len(manifest["rows"]) == 20
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as dest:
    with zipfile.ZipFile(REPO / "verify_bundle.zip") as source:
        for name in source.namelist():
            if not name.startswith("data/"):
                dest.writestr(name, source.read(name))
    for name in ("manifest.json", "labels.csv", "LICENSE", "ANNOTATIONS"):
        dest.write(DATA / name, "pilot/" + name)
    for row in manifest["rows"]:
        path = DATA / "test" / (row["ID"] + ".wav")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["clip_sha256"]
        dest.write(path, "pilot/test/" + path.name)
    dest.writestr("pilot/ATTRIBUTION.txt", "MUSAN: David Snyder, Guoguo Chen, Daniel Povey (2015), https://www.openslr.org/17/ . Individual artists/titles/terms: LICENSE and ANNOTATIONS.\nFakeMusicCaps: Politecnico di Milano ISPL, https://zenodo.org/records/15063698 ; CC BY-NC 4.0 https://creativecommons.org/licenses/by-nc/4.0/ .\nChanges: centered 8-second crops, mono 16kHz PCM16. Internal noncommercial diagnostic; not training.\n")
bundle_hash = hashlib.sha256(OUT.read_bytes()).hexdigest()
verified = json.loads((REPO / "notebooks/colab_verify.ipynb").read_text(encoding="utf-8"))
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add("markdown", """
# 진짜·가짜 음악 파일 비교 — 첫 소규모 진단
**대회 제출용이 아닙니다.** `music_pilot_bundle.zip`만 업로드합니다.
진짜 10개와 생성 음악 10개를 모두 같은 길이·형식으로 맞췄습니다.
학습하지 않고, 현재 코드에서 미리 정한 세 설정의 FILE EER/AUC를 비교합니다.
음성·음악 성분 라벨이 없으므로 Music EER, Voice EER, CPS, 대회 총점은 계산하지 않습니다.
작은 출처 편향 표본이며 검증셋이나 최종 홀드아웃을 대체하지 않습니다.
점수가 좋아도 이 결과만으로 제출하거나 일반화 성능이 개선됐다고 판단하지 않습니다.

GPU를 선택하고 위에서부터 실행하세요. 마지막에 `music_pilot_results.zip`이 내려받아집니다.
오류가 나면 출력을 포함한 ipynb를 저장해 보내세요. 비용이 드는 업그레이드는 필요 없습니다.
""")
add("code", """
import torch, sys, platform
assert torch.cuda.is_available(), "런타임 → 런타임 유형 변경 → GPU를 선택하세요."
print(torch.cuda.get_device_name(0), torch.__version__, sys.version)
print("무료 GPU 환경의 진단입니다. L4 제출 서버 재현 검증은 별도입니다.")
""")
add("markdown", "## 1. music_pilot_bundle.zip 올리기")
add("code", f"""
from google.colab import files
from pathlib import Path
import hashlib, zipfile, tempfile, json, shutil, os
uploaded = files.upload()
assert len(uploaded) == 1, "music_pilot_bundle.zip 하나만 선택하세요."
blob = next(iter(uploaded.values()))
assert hashlib.sha256(blob).hexdigest() == "{bundle_hash}", "다른 버전의 ZIP입니다. 새 실험 ZIP을 선택하세요."
WORK = Path(tempfile.mkdtemp(prefix="deepvoice_pilot_", dir="/content"))
import io
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for name in z.namelist():
        assert (WORK / name).resolve().is_relative_to(WORK.resolve())
    z.extractall(WORK)
RESULTS = WORK / "results"
RESULTS.mkdir()
manifest = json.loads((WORK / "pilot/manifest.json").read_text())
for row in manifest["rows"]:
    path = WORK / "pilot/test" / (row["ID"] + ".wav")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["clip_sha256"]
print("실험 음원 20개 무결성 확인 완료", WORK)
""")
add("markdown", "## 2. 이전 실행에서 사용한 패키지 설치 / 버전 기록")
install = ''.join(verified["cells"][6]["source"])
add("code", install + '\n(RESULTS / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True) + "\\n" + sys.version + "\\n" + torch.cuda.get_device_name(0))')
add("markdown", "## 3. 모델 준비 — 이전 Colab 파일이 남아 있으면 재사용")
model_code = ''.join(verified["cells"][8]["source"]).replace('WORK = pathlib.Path("/content/submit")', '# WORK는 이번 실험 전용 폴더')
add("code", """
import pathlib, hashlib, shutil, os
MODEL = WORK / "model"
def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()
expected = dict(line.split() for line in (MODEL / "SHA256SUMS.txt").read_text().splitlines() if line.strip())
# 위 파일은 digest -> relative path 순서다.
for checksum, relative in expected.items():
    previous = Path("/content/submit/model") / relative
    target = MODEL / relative
    if not target.exists() and previous.is_file() and digest(previous) == checksum:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(previous, target)
        except OSError:
            shutil.copy2(previous, target)
have_all = all((MODEL / rel).is_file() and digest(MODEL / rel) == checksum for checksum, rel in expected.items())
""" + '\nif not have_all:\n' + textwrap.indent(model_code, '    ') + '\nfor line in (MODEL / "SHA256SUMS.txt").read_text().splitlines():\n    if line.strip():\n        checksum, relative = line.split()\n        assert digest(MODEL / relative) == checksum, relative\nprint("모델 해시 검사 통과")')
add("markdown", """
## 4. 정답·입력 준비
정답은 공개 데이터의 생성 여부에서 가져왔습니다. MUSAN은 실제 녹음, FakeMusicCaps는 생성 모델 출력입니다.
직접 청취로 성분 라벨을 검토한 데이터가 아니므로 파일 생성 여부만 진단합니다.
진짜 표본은 아카이브 앞부분이라 같은 연주자가 여러 번 포함됩니다. 파일 수만큼 독립 표본이 있다고 해석하지 마세요.
""")
add("code", """
import pandas as pd, numpy as np, soundfile as sf
labels = pd.read_csv(WORK / "pilot/labels.csv", dtype={"ID": str})
assert not labels.ID.duplicated().any() and labels.file_fake.value_counts().to_dict() == {0: 10, 1: 10}
shutil.copytree(WORK / "pilot/test", WORK / "data/test")
columns = ["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]
template = pd.DataFrame({"ID": labels.ID})
for column in columns[1:]:
    template[column] = 0.5
template.to_csv(WORK / "data/sample_submission.csv", index=False)
for path in (WORK / "data/test").glob("*.wav"):
    audio, sr = sf.read(path)
    assert sr == 16000 and audio.shape == (128000,) and np.isfinite(audio).all()
print(labels.file_fake.value_counts().rename(index={0: "진짜", 1: "생성"}))
shutil.copy2(WORK / "pilot/manifest.json", RESULTS / "manifest.json")
shutil.copy2(WORK / "pilot/labels.csv", RESULTS / "labels.csv")
""")
add("markdown", """
## 5. 세 설정을 비교하고 결과 다운로드
설정 1: 현재 direct + max. 설정 2: direct + mean (모든 성분 집계 변경).
설정 3: baseline fusion + max. 다른 설정과 가중치는 동일합니다.
각 설정은 모델을 한 번 로드하고 20개 파일을 연속 처리합니다.
실행 전체 시간을 저장하지만 1,200개 파일 시간으로 환산하지 않습니다.
**EER은 낮을수록, AUC는 높을수록 좋습니다. 작은 표본의 진단값이며 대회 점수 예측이 아닙니다.**
""")
add("code", r'''
import ast, time, subprocess, re, traceback
from sklearn.metrics import roc_curve, roc_auc_score
source = (WORK / "script.py").read_text()
tree = ast.parse(source)
node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
base = ast.literal_eval(node.value)
variants = [("direct_max", {"file_head": "direct", "segment_agg": "max"}),
            ("direct_mean", {"file_head": "direct", "segment_agg": "mean"}),
            ("fusion_max", {"file_head": "fusion", "segment_agg": "max", "fusion_mode": "baseline"})]
summary = []
env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
(RESULTS / "baseline_script.py").write_text(source)
try:
    for name, changes in variants:
        cfg = dict(base, **changes)
        lines = source.splitlines(keepends=True)
        edited = ''.join(lines[:node.lineno-1]) + "CONFIG = " + repr(cfg) + "\n" + ''.join(lines[node.end_lineno:])
        ast.parse(edited)
        (WORK / "script.py").write_text(edited)
        (RESULTS / (name + "_config.json")).write_text(json.dumps(cfg, indent=2))
        prediction_path = WORK / "output/submission.csv"
        prediction_path.unlink(missing_ok=True)
        started = time.perf_counter()
        run = subprocess.run([sys.executable, "script.py"], cwd=WORK, env=env, capture_output=True, text=True, timeout=1800)
        elapsed = time.perf_counter() - started
        log = run.stdout + run.stderr
        (RESULTS / (name + ".log")).write_text(log)
        print(name, "elapsed", round(elapsed, 1), "returncode", run.returncode)
        print(log[-2000:])
        run.check_returncode()
        assert "실패 0건" in log and "폴백값 사용" not in log, "일부 파일 처리 실패: 로그 확인"
        pred = pd.read_csv(prediction_path, dtype={"ID": str})
        assert list(pred.columns) == columns and len(pred) == len(labels)
        assert not pred.ID.duplicated().any() and set(pred.ID) == set(labels.ID)
        values = pred[columns[1:]].to_numpy(dtype=float)
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        merged = labels.merge(pred, on="ID", validate="one_to_one")
        truth, score = merged.file_fake.to_numpy(), merged.FILE_FAKE_PROB.to_numpy()
        fpr, tpr, _ = roc_curve(truth, score, pos_label=1, drop_intermediate=False)
        fnr = 1-tpr
        index = np.argmin(np.abs(fpr-fnr))
        summary.append({"variant": name, "file_eer": float((fpr[index]+fnr[index])/2),
                        "file_auc": float(roc_auc_score(truth, score)), "process_seconds": elapsed,
                        "script_sha256": hashlib.sha256(edited.encode()).hexdigest(), "n": len(truth)})
        merged.to_csv(RESULTS / (name + "_predictions.csv"), index=False)
        pd.DataFrame(summary).to_csv(RESULTS / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
except Exception:
    (RESULTS / "ERROR.txt").write_text(traceback.format_exc())
    raise
finally:
    (WORK / "script.py").write_text(source)
    if Path("/content/install.log").exists():
        shutil.copy2("/content/install.log", RESULTS / "install.log")
    output_zip = shutil.make_archive("/content/music_pilot_results", "zip", RESULTS)
    files.download(output_zip)
''')
notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
             "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 0}
target = REPO / "notebooks/colab_music_pilot.ipynb"
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(target, OUT.stat().st_size, bundle_hash)
