#!/usr/bin/env python3
"""notebooks/colab_verify.ipynb 를 생성한다.

노트북 JSON 을 손으로 쓰면 깨지기 쉬워서 스크립트로 만든다.
내용을 고칠 때는 이 파일을 고치고 다시 실행한다.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "colab_verify.ipynb"

DF_ARENA_REPO = "Speech-Arena-2025/DF_Arena_1B_V_1"
DF_ARENA_REV = "fb6ce85de12c2c5a509d89114adaf827dd75f49f"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)})


def code(text):
    CELLS.append({
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.strip().splitlines(True),
    })


md("""
# Deep-voice 제출 검증 (Colab)

로컬에 GPU 가 없어 리더보드가 첫 테스트가 되는 상황을 피하기 위한 노트북이다.
**런타임 유형을 GPU 로 바꾸고 실행한다.** 평가 서버는 L4 이므로 L4 를 고르면 가장 정확하다.

확인하는 것:

1. `script.py` 가 끝까지 도는가 (제출 오류 = 하루 3회 중 1회 소모)
2. DF-Arena 배치 패치가 단일 경로와 같은 값을 내는가
3. MP3 / FLAC / 8kHz 전화채널을 디코딩하는가 (배포 더미는 WAV 뿐이라 여기서 사고가 난다)
4. 1,200 파일 60분 안에 들어가는가

가중치는 업로드하지 않는다. 4.4GB 는 대회 배포본과 **동일한 HF 리비전**에서 받고
SHA-256 으로 대조한다.
""")

md("## 0. GPU 확인")
code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("평가 서버: NVIDIA L4 22.4GiB / torch 2.7.1+cu128 / CUDA 12.8 / Python 3.11.15")
""")

md("""
## 1. verify_bundle.zip 업로드

로컬 `Deep-voice/verify_bundle.zip` (약 250KB) 을 올린다.
`scripts/build_submit.py` 가 아니라 `scripts/make_verify_bundle.py` 로 만든 것이다.
""")
code("""
from google.colab import files
import zipfile, pathlib, shutil

WORK = pathlib.Path("/content/submit")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

uploaded = files.upload()          # verify_bundle.zip 선택
name = next(iter(uploaded))
with zipfile.ZipFile(name) as z:
    z.extractall(WORK)

for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"{p.stat().st_size/1024:8.1f} KB  {p.relative_to(WORK)}")
""")

md("## 2. 평가 서버와 같은 패키지 설치")
code("""
# 평가 서버 기본 설치 목록과 맞춘다. Colab 은 torch 가 이미 있으므로 건드리지 않는다.
!pip -q install demucs==4.0.1 panns-inference==0.1.1 "librosa==0.10.2.post1" soundfile soxr 2>&1 | tail -3
import importlib
for mod in ["librosa", "demucs", "panns_inference", "transformers", "torchaudio"]:
    m = importlib.import_module(mod)
    print(mod, getattr(m, "__version__", "?"))
""")

md("""
## 3. 대용량 가중치 확보 + 무결성 대조

대회 배포본의 `SHA256SUMS.txt` 와 대조한다. 일치해야 이 검증이 의미가 있다.
""")
code(f"""
import hashlib, pathlib, shutil, os
from huggingface_hub import snapshot_download

WORK = pathlib.Path("/content/submit")
MODEL = WORK / "model"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()

expected = {{}}
for line in (MODEL / "SHA256SUMS.txt").read_text().splitlines():
    if line.strip():
        digest, rel = line.split()
        expected[rel] = digest

# --- DF-Arena 1B (대회 배포본과 동일 리비전) ---
snap = snapshot_download(repo_id="{DF_ARENA_REPO}", revision="{DF_ARENA_REV}",
                         allow_patterns=["pytorch_model.bin"])
shutil.copy2(pathlib.Path(snap) / "pytorch_model.bin", MODEL / "df_arena_1b" / "pytorch_model.bin")

# --- HTDemucs (demucs 가 캐시에 받아둔 .th 를 로컬 repo 로 옮긴다) ---
from demucs.pretrained import get_model as _dl
import torch
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{{**k, "weights_only": False}})
try:
    _dl("htdemucs")
finally:
    torch.load = _orig
cache = pathlib.Path(torch.hub.get_dir()) / "checkpoints"
for th in cache.glob("*.th"):
    shutil.copy2(th, MODEL / "htdemucs" / th.name)

# --- PANNs Cnn14 (panns_inference 가 ~/panns_data 에 받는다) ---
from panns_inference import AudioTagging
AudioTagging(checkpoint_path=None, device="cpu")
shutil.copy2(pathlib.Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth",
             MODEL / "panns" / "Cnn14_mAP=0.431.pth")

print()
ok = True
for rel, digest in expected.items():
    path = MODEL / rel
    if not path.exists():
        print(f"[FAIL] 없음 {{rel}}"); ok = False; continue
    got = sha256(path)
    mark = "OK  " if got == digest else "FAIL"
    if got != digest:
        ok = False
    print(f"[{{mark}}] {{rel}}  {{got[:16]}}...")
print("\\n무결성", "일치 — 대회 배포본과 같은 가중치다" if ok else "불일치! 이 검증은 신뢰할 수 없다")
""")

md("""
## 4. 포맷 내성 시험 데이터 생성

배포 더미는 WAV 3개(각 3.405초)뿐이다. 실제 평가 데이터는 **MP3·FLAC 혼재, 모노/스테레오 혼재,
일부 전화채널, 길이 4초~1분**이다. 여기서 디코딩이 깨지면 그대로 0점이므로 미리 만들어 시험한다.

시간 측정용이기도 하다 — 길이별 처리 속도를 알아야 1,200개를 환산할 수 있다.
""")
code("""
import subprocess, pathlib, numpy as np, soundfile as sf, librosa, csv

WORK = pathlib.Path("/content/submit")
STRESS = pathlib.Path("/content/stress")
if STRESS.exists():
    import shutil; shutil.rmtree(STRESS)
(STRESS / "test").mkdir(parents=True)

base, sr = librosa.load(WORK / "data" / "test" / "TEST_0000.wav", sr=16000, mono=True)

def make(seconds):
    reps = int(np.ceil(seconds * sr / base.size))
    return np.tile(base, reps)[: int(seconds * sr)].astype(np.float32)

specs = []          # (id, 길이초, 확장자, 설명)
for seconds in (4, 15, 30, 60):
    specs += [(f"S_{seconds:02d}_wav",  seconds, "wav",  "16k mono wav"),
              (f"S_{seconds:02d}_mp3",  seconds, "mp3",  "mp3 128k"),
              (f"S_{seconds:02d}_flac", seconds, "flac", "flac"),
              (f"S_{seconds:02d}_tel",  seconds, "wav",  "8k 전화채널 -> 16k")]
specs += [("S_stereo", 10, "wav", "스테레오"), ("S_short", 4, "wav", "정확히 4.00초")]

for audio_id, seconds, ext, note in specs:
    audio = make(seconds)
    out = STRESS / "test" / f"{audio_id}.{ext}"
    if note.startswith("8k"):
        tmp = STRESS / "tmp.wav"; sf.write(tmp, audio, sr)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp),
                        "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le", str(out)], check=True)
    elif note == "스테레오":
        sf.write(out, np.stack([audio, audio * 0.7], axis=1), sr)
    elif ext == "mp3":
        tmp = STRESS / "tmp.wav"; sf.write(tmp, audio, sr)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp),
                        "-b:a", "128k", str(out)], check=True)
    else:
        sf.write(out, audio, sr)

with (STRESS / "sample_submission.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"])
    for audio_id, *_ in specs:
        w.writerow([audio_id, 0.5, 0.5, 0.5, 0.5, 0.5])

for p in sorted((STRESS / "test").iterdir()):
    print(f"{p.stat().st_size/1024:8.1f} KB  {p.name}")
""")

md("""
## 5. 배포 더미 3개로 1차 실행

먼저 원래 더미로 돌려 끝까지 도는지 본다. 로그에서 확인할 것:

- `배치 경로 검증 통과 (최대 편차 ...)` — 배치 패치가 단일 경로와 같은 값을 내는가
- `bf16 vs fp32 최대 편차` — bf16 을 써도 되는가
- `실패 N건` — 0 이어야 한다
""")
code("""
import subprocess, pathlib, shutil
WORK = pathlib.Path("/content/submit")

result = subprocess.run(["python", "script.py"], cwd=WORK, capture_output=True, text=True)
print(result.stdout[-6000:])
if result.returncode != 0:
    print("=== STDERR ===")
    print(result.stderr[-6000:])
print("\\nreturncode:", result.returncode)
""")
code("""
import pandas as pd, pathlib
df = pd.read_csv(pathlib.Path("/content/submit/output/submission.csv"))
print(df.to_string(index=False))
assert df.shape == (3, 6), df.shape
values = df.iloc[:, 1:]          # DataFrame 에는 between() 이 없다. Series 메서드다.
assert ((values >= 0) & (values <= 1)).all().all(), "확률이 [0,1] 범위를 벗어났다"
assert values.notna().all().all(), "결측값이 있다"
assert list(df.columns) == ["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
                            "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"], list(df.columns)

# 융합식이 CONFIG["fusion_mode"]="baseline" 대로 계산됐는지 대조한다
expected = (values["VOICE_PRESENT_PROB"] * values["VOICE_FAKE_PROB"]).combine(
    values["MUSIC_PRESENT_PROB"] * values["MUSIC_FAKE_PROB"], max)
assert (expected - values["FILE_FAKE_PROB"]).abs().max() < 1e-6, "FILE_FAKE 융합식 불일치"
print("\\n형식 검사 통과")
""")

md("""
## 6. 포맷 내성 + 시간 측정

4절에서 만든 세트로 돌린다. **여기서 실패가 0이 아니면 실제 평가에서도 실패한다.**
""")
code("""
import subprocess, pathlib, shutil, time, re
WORK = pathlib.Path("/content/submit")

backup = pathlib.Path("/content/data_backup")
if backup.exists(): shutil.rmtree(backup)
shutil.move(str(WORK / "data"), str(backup))
shutil.copytree("/content/stress", WORK / "data")

started = time.time()
result = subprocess.run(["python", "script.py"], cwd=WORK, capture_output=True, text=True)
elapsed = time.time() - started

print(result.stdout[-8000:])
if result.returncode != 0:
    print("=== STDERR ==="); print(result.stderr[-4000:])

shutil.rmtree(WORK / "data")
shutil.move(str(backup), str(WORK / "data"))
print(f"\\n총 {elapsed:.1f}s, returncode {result.returncode}")
""")

md("""
## 7. 1,200 파일 환산

파일 수가 아니라 **오디오 길이당 처리 속도**로 환산해야 한다.
평가 데이터는 4초~1분이고 평균값은 공개되지 않았으므로 여러 시나리오로 본다.
""")
code("""
import subprocess, pathlib, shutil, time, numpy as np, librosa
WORK = pathlib.Path("/content/submit")
STRESS = pathlib.Path("/content/stress/test")

# 길이별 1개씩만 골라 개별 측정한다
import csv, tempfile
durations = [4, 15, 30, 60]
per_second = []

for seconds in durations:
    src = STRESS / f"S_{seconds:02d}_wav.wav"
    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "test").mkdir()
    shutil.copy2(src, tmp / "test" / "T.wav")
    with (tmp / "sample_submission.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ID","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB",
                    "VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"])
        w.writerow(["T",0.5,0.5,0.5,0.5,0.5])

    backup = pathlib.Path("/content/db2")
    if backup.exists(): shutil.rmtree(backup)
    shutil.move(str(WORK / "data"), str(backup))
    shutil.copytree(tmp, WORK / "data")

    t0 = time.time()
    r = subprocess.run(["python","script.py"], cwd=WORK, capture_output=True, text=True)
    total = time.time() - t0

    # 모델 로드 시간을 로그에서 빼서 순수 추론 시간을 얻는다
    load = 0.0
    for line in r.stdout.splitlines():
        if "모델 로드" in line:
            load = float(line.split()[-1].rstrip("s"))
    infer = max(total - load, 0.01)
    per_second.append(infer / seconds)
    print(f"{seconds:3d}초 파일: 전체 {total:6.1f}s, 로드 {load:5.1f}s, 추론 {infer:6.2f}s "
          f"({infer/seconds:.3f} s/오디오초)")

    shutil.rmtree(WORK / "data"); shutil.move(str(backup), str(WORK / "data"))

rate = float(np.median(per_second))
print(f"\\n중앙값 처리 속도: {rate:.3f} 초 / 오디오 1초")
print("\\n1,200 파일 환산 (60분 = 3600초 한계):")
for avg in (10, 15, 20, 25, 30, 40):
    est = rate * avg * 1200
    mark = "OK" if est < 3600 * 0.7 else ("빠듯" if est < 3600 else "초과")
    print(f"  평균 {avg:2d}초 -> {est/60:6.1f}분   [{mark}]")
""")

md("""
## 8. 결과 정리

아래를 `docs/04_experiment-log.md` 에 옮겨 적는다.

- 배치 패치 편차, bf16 편차 → 켜도 되는지 판단 근거
- 포맷별 실패 여부 → 실패가 있으면 그것부터 고친다
- 처리 속도와 1,200 파일 환산 → 60분에 여유가 있으면 **정확도에 재투자**한다
  (속도는 점수가 아니다. CPS 는 존재 탐지 점수다)

여유가 없으면 `CONFIG["demucs_gating"] = True` 를 켜서 분리 횟수를 줄인다.
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "L4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{OUT} 생성 — 셀 {len(CELLS)}개")
