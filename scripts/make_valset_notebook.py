#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""notebooks/kaggle_valset.ipynb 를 생성한다.

노트북 JSON 을 손으로 쓰면 깨지기 쉬워서 스크립트로 만든다.
내용을 고칠 때는 이 파일을 고치고 다시 실행한다.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "kaggle_valset.ipynb"

DF_ARENA_REPO = "Speech-Arena-2025/DF_Arena_1B_V_1"
DF_ARENA_REV = "fb6ce85de12c2c5a509d89114adaf827dd75f49f"

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip().splitlines(True)})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip().splitlines(True)})


md("""
# 로컬 검증셋 + 설정 스윕 (Kaggle)

리더보드는 하루 3회뿐이다. 설정 비교를 여기서 전부 하고 **이긴 것만 제출**한다.

## 설계

```
[1회, GPU]  파일마다 PANNs(vp, mp) + DF-Arena 세그먼트 점수 3종
            (원본 / 음성 스템 / 음악 스템) 을 캐시에 담는다
[무한, CPU] gating · file_head · fusion_mode · segment_agg · 게이트 임계값 조합을
            캐시 위에서 즉시 평가한다
```

스윕은 `submit/script.py` 의 `process_one_file` 을 **그대로 호출**하고 채점기만 캐시로 바꾼다.
로직을 베껴 쓰지 않으므로 실제 추론과 어긋날 수 없다.

## 준비 — 이 노트북을 돌리기 전에

1. 우상단 **Settings → Accelerator → GPU** (P100 또는 T4 x2)
2. **Settings → Internet → On** (모델 가중치를 받는다)
3. 오른쪽 **+ Add Input** 으로 데이터셋을 붙인다:
   - `mohammedabdeldayem/the-fake-or-real-dataset` — 실제/가짜 **음성**
   - 실제 **음악** (MUSAN music 또는 FMA small 계열 아무거나)
   - (선택) 가짜 **음악** — 없으면 Music EER 은 계산되지 않는다
4. 로컬 `Deep-voice/verify_bundle.zip` 을 **+ Add Input → Upload** 로 올린다

주의: 대회 평가 데이터는 절대 여기에 올리지 않는다. 검증셋은 전부 공개 데이터로 합성한다.
""")

md("## 0. 환경 확인")
code("""
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "Settings -> Accelerator -> GPU 로 바꿔라"

import glob
print("\\n[붙어 있는 입력]")
for path in sorted(glob.glob("/kaggle/input/*")):
    print(" ", path)
""")

md("""
## 1. 코드 번들 풀기

`verify_bundle.zip` 을 올린 입력 폴더를 자동으로 찾는다.
""")
code("""
import glob, zipfile, pathlib, shutil, sys

WORK = pathlib.Path("/kaggle/working/submit")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

candidates = glob.glob("/kaggle/input/**/verify_bundle.zip", recursive=True)
assert candidates, "verify_bundle.zip 을 입력으로 올려라"
print("번들:", candidates[0])
with zipfile.ZipFile(candidates[0]) as z:
    z.extractall(WORK)

sys.path.insert(0, str(WORK / "src"))
print("\\n[번들 내용]")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size/1024:8.1f} KB  {p.relative_to(WORK)}")
""")

md("""
## 2. 소스 오디오 수집

FoR 데이터셋은 배포본마다 폴더 구조가 다르다. 아래는 `real` / `fake` 라는 이름의 폴더를
찾아 그 안의 오디오를 모은다. 잘 안 잡히면 `MANUAL` 에 경로를 직접 적는다.
""")
code("""
import glob, pathlib, random

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
MAX_PER_SOURCE = 4000          # 너무 많으면 수집만 오래 걸린다

# 자동 탐색이 실패하면 여기에 직접 적는다. 예: {"voice_real": ["/kaggle/input/.../real"]}
MANUAL = {}

def collect(root, limit=MAX_PER_SOURCE):
    out = []
    for path in pathlib.Path(root).rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXT:
            out.append(path)
            if len(out) >= limit:
                break
    return out

def autodetect():
    found = {"voice_real": [], "voice_fake": [], "music_real": [], "music_fake": []}
    for base in sorted(glob.glob("/kaggle/input/*")):
        low = base.lower()
        is_music = any(k in low for k in ("music", "musan", "fma", "song", "audioset"))
        for sub in pathlib.Path(base).rglob("*"):
            if not sub.is_dir():
                continue
            name = sub.name.lower()
            if name in ("real", "bonafide", "genuine", "original"):
                key = "music_real" if is_music else "voice_real"
            elif name in ("fake", "spoof", "synthetic", "generated"):
                key = "music_fake" if is_music else "voice_fake"
            else:
                continue
            if len(found[key]) < MAX_PER_SOURCE:
                found[key].extend(collect(sub, MAX_PER_SOURCE - len(found[key])))
    # real/fake 폴더 이름이 없는 음악 데이터셋은 전체를 실제 음악으로 본다
    if not found["music_real"]:
        for base in sorted(glob.glob("/kaggle/input/*")):
            low = base.lower()
            if any(k in low for k in ("music", "musan", "fma")) and "fake" not in low:
                found["music_real"].extend(collect(base))
                break
    return found

SOURCES = autodetect()
for key, paths in MANUAL.items():
    SOURCES[key] = [p for root in paths for p in collect(root)]

print("[수집 결과]")
for key in ("voice_real", "voice_fake", "music_real", "music_fake"):
    paths = SOURCES.get(key, [])
    print(f"  {key:12s} {len(paths):6d}개  {paths[0] if paths else '(없음)'}")

missing = [k for k in ("voice_real", "voice_fake", "music_real") if not SOURCES.get(k)]
assert not missing, f"필수 소스가 비었다: {missing} — MANUAL 에 경로를 직접 적어라"
if not SOURCES.get("music_fake"):
    print("\\n[warn] 가짜 음악 소스가 없다 -> Music EER 은 nan 이 된다 (실효 가중치 0.27 미측정)")
""")

md("""
## 3. 검증셋 합성

평가 데이터의 성질을 재현한다 — 음성/음악/혼합, 4~60초, MP3·FLAC·WAV, 모노/스테레오,
일부 전화채널. 그리고 **REAL 에 후처리를 건 샘플**을 일부러 넣는다
(대회 규정상 후처리만으로는 FAKE 가 아니다).
""")
code("""
import importlib, sys
sys.path.insert(0, "/kaggle/working/submit/src")
from synth import build_valset
importlib.reload(build_valset)

VALSET_DIR = "/kaggle/working/valset"
N_CLIPS = 400          # 먼저 400 으로 확인하고, 시간이 되면 늘린다

rows = build_valset.build(SOURCES, VALSET_DIR, count=N_CLIPS, seed=0)
""")

md("""
## 4. 모델 준비

대회 배포본과 **동일한 HF 리비전**에서 받고 SHA-256 으로 대조한다.
""")
code(f"""
import hashlib, pathlib, shutil, torch
from huggingface_hub import snapshot_download

WORK = pathlib.Path("/kaggle/working/submit")
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

snap = snapshot_download(repo_id="{DF_ARENA_REPO}", revision="{DF_ARENA_REV}",
                         allow_patterns=["pytorch_model.bin"])
shutil.copy2(pathlib.Path(snap) / "pytorch_model.bin", MODEL / "df_arena_1b" / "pytorch_model.bin")

from demucs.pretrained import get_model as _dl
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{{**k, "weights_only": False}})
try:
    _dl("htdemucs")
finally:
    torch.load = _orig
for th in (pathlib.Path(torch.hub.get_dir()) / "checkpoints").glob("*.th"):
    shutil.copy2(th, MODEL / "htdemucs" / th.name)

from panns_inference import AudioTagging
AudioTagging(checkpoint_path=None, device="cpu")
shutil.copy2(pathlib.Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth",
             MODEL / "panns" / "Cnn14_mAP=0.431.pth")

ok = True
for rel, digest in expected.items():
    got = sha256(MODEL / rel)
    ok &= (got == digest)
    print(f"[{{'OK  ' if got == digest else 'FAIL'}}] {{rel}}")
print("\\n무결성", "일치 — 대회 배포본과 같은 가중치다" if ok else "불일치! 신뢰할 수 없다")
assert ok
""")

md("""
## 5. 캐시 생성 (GPU, 이 노트북에서 가장 오래 걸린다)

게이팅 임계값까지 스윕하려면 **모든 파일의 스템 점수**가 필요하므로 여기서는 항상 분리한다.
실제 추론보다 느리지만 한 번만 하면 된다.
""")
code("""
import csv, pathlib, sys, torch
sys.path.insert(0, "/kaggle/working/submit/src")
from eval import sweep as sweep_mod

script = sweep_mod.load_script("/kaggle/working/submit/script.py")
device = torch.device("cuda")

panns = script.load_panns_model(device)
scorer = script.DFArenaScorer(device)
htdemucs = script.load_htdemucs_model(device)

with open("/kaggle/working/valset/labels.csv", encoding="utf-8") as f:
    labels = list(csv.DictReader(f))
print(f"검증셋 {len(labels)}개")

# want_embeddings=True 로 두면 음악 프로브 학습용 임베딩까지 같이 모은다 (메모리 증가)
cache = sweep_mod.precompute(script, labels, "/kaggle/working/valset/test",
                             panns, scorer, htdemucs, device, want_embeddings=False)
""")

code("""
# 캐시를 저장해두면 이후 스윕은 GPU 없이도 돌릴 수 있다
import pickle, pathlib
with open("/kaggle/working/cache.pkl", "wb") as f:
    pickle.dump({"cache": cache, "labels": labels}, f)
print("cache.pkl", pathlib.Path("/kaggle/working/cache.pkl").stat().st_size / 1024**2, "MB")
""")

md("""
## 6. 설정 스윕

여기부터는 GPU 를 쓰지 않는다. 조합을 얼마든지 늘려도 된다.
""")
code("""
results = sweep_mod.sweep(script, cache, labels, sweep_mod.default_grid(),
                          baseline_name="baseline(fusion)")
""")

md("""
## 7. 2단계 — 1위 설정 위에 추가 변형을 얹는다

1단계에서 이긴 설정을 고정하고 그 위에서 다시 스윕한다.
""")
code("""
best = results[0]
print("1단계 1위:", best["name"], best["config"], f"Score {best['score']:.5f}\\n")

extra = [
    ("+agg:topk2", {"segment_agg": "topk_mean", "segment_topk": 2}),
    ("+agg:mean", {"segment_agg": "mean"}),
    ("+music_agg:mean", {"segment_agg_music": "mean"}),
    ("+gate:m0.02", {"gate_music": 0.02}),
    ("+gate:m0.20", {"gate_music": 0.20}),
    ("+gate:v0.40", {"gate_voice": 0.40}),
    ("+gate:off", {"demucs_gating": False}),
]
stage2 = [(best["name"], best["config"])] + sweep_mod.cross_grid(best["config"], extra)
results2 = sweep_mod.sweep(script, cache, labels, stage2, baseline_name=best["name"])
""")

md("""
## 8. 결론

최종 1위 설정을 `submit/script.py` 의 `CONFIG` 에 옮겨 적고 `build_submit.py` 로 다시 빌드한다.

읽는 법:

- **Score 차이가 0.005 미만이면 노이즈로 본다.** 검증셋 400개에서 EER 은 표본 오차가 크다.
- 검증셋은 **합성 근사**다. 리더보드와 순위가 다를 수 있다.
  그래도 "명백히 나쁜 설정을 공짜로 걸러낸다"는 목적에는 충분하다.
- `music_eer` 이 nan 이면 가짜 음악 소스가 없었다는 뜻이다. 그 열은 판단에서 제외한다.
""")
code("""
print("최종 순위")
for rank, r in enumerate(results2, 1):
    print(f"{rank:>2}. {r['name']:<28} Score {r['score']:.5f}  "
          f"(file {r['file_eer']:.4f} / voice {r['voice_eer']:.4f} / music {r['music_eer']:.4f})")
print("\\n최고 설정 CONFIG 덮어쓰기:")
print(results2[0]["config"])
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{OUT} 생성 — 셀 {len(CELLS)}개")
