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
## 0-1. 패키지 설치

Kaggle 기본 이미지에는 `demucs` 와 `panns-inference` 가 없다.
`torch` 는 이미 있으므로 건드리지 않는다 (demucs 4.0.1 은 `torch>=1.8.1` 만 요구한다).
""")
code("""
!pip -q install demucs==4.0.1 panns-inference==0.1.1 2>&1 | tail -5

import importlib, torch
for mod in ("demucs", "panns_inference", "librosa", "soundfile", "torchaudio"):
    try:
        m = importlib.import_module(mod)
        print(f"  OK   {mod:18s} {getattr(m, '__version__', '')}")
    except Exception as error:
        print(f"  FAIL {mod:18s} {error}")

print("\\ntorch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "설치 과정에서 torch 가 바뀌었다 — Run > Restart Session 후 처음부터"
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

# Kaggle 은 업로드한 zip 을 데이터셋으로 만들 때 자동으로 풀어버린다.
# zip 이 그대로 있는 경우와 이미 풀린 경우를 모두 처리한다.
zips = glob.glob("/kaggle/input/**/verify_bundle.zip", recursive=True)
if zips:
    print("zip 형태로 발견:", zips[0])
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORK)
else:
    # 풀린 형태 찾기 — script.py 옆에 model/ 또는 src/ 가 있는 폴더가 번들 루트다
    found_roots = [f.parent for f in pathlib.Path("/kaggle/input").rglob("script.py")
                   if (f.parent / "model").is_dir() or (f.parent / "src").is_dir()]
    if not found_roots:
        print("[진단] /kaggle/input 아래 내용:")
        for entry in sorted(pathlib.Path("/kaggle/input").rglob("*")):
            if len(entry.relative_to("/kaggle/input").parts) <= 3:
                print("  ", entry)
        raise AssertionError(
            "번들을 못 찾았다. deepvoice-bundle 을 Add Input 으로 붙였는지 확인하라")
    print("풀린 형태로 발견:", found_roots[0])
    shutil.copytree(found_roots[0], WORK, dirs_exist_ok=True)

sys.path.insert(0, str(WORK / "src"))
missing_files = [n for n in ("script.py", "src/eval/sweep.py",
                             "src/synth/build_valset.py",
                             "model/panns/component_labels.json")
                 if not (WORK / n).exists()]
assert not missing_files, f"번들에서 빠진 파일: {missing_files}"
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

MUSIC_WORDS = ("music", "musan", "fma", "gtzan", "genre", "song", "instrument")

# 생성 음악 데이터셋·생성기 이름. 이 목록에 걸리면 폴더 구조와 무관하게 music_fake 다.
# FakeMusicCaps 는 fake/ 폴더가 없고 생성기 이름으로 나뉘어 있어서, real/fake 하위폴더
# 규칙만으로는 한 개도 못 잡는다.
MUSIC_FAKE_WORDS = ("fakemusiccaps", "musicgen", "musicldm", "audioldm",
                    "stableaudio", "stable_audio", "mustango",
                    "suno", "udio", "sonics", "ai-music", "ai_music")

# 생성 음악 데이터셋 안에 들어 있는 진짜 음악 하위셋. FakeMusicCaps 는 MusicCaps 원본을
# 같이 담고 있으므로 이걸 fake 로 넣으면 라벨이 통째로 뒤집힌다.
MUSIC_REAL_INSIDE_FAKE = ("musiccaps", "real", "bonafide", "original")

# 보컬이 섞여 나오는 상용 생성기. 대회 정의상 음악은 **보컬 없는 반주·악기음만**이므로
# (docs/00_competition-spec.md:66) 노래가 든 클립을 music_fake 로 쓰면
# VOICE_PRESENT 라벨이 틀어지고 Voice EER 까지 오염된다. Phase 2 에서 PANNs 로 거른다.
VOCAL_RISK_WORDS = ("suno", "udio")


def dataset_roots():
    # Kaggle 입력 경로가 두 가지다:
    #   예전  /kaggle/input/<이름>/
    #   지금  /kaggle/input/datasets/<소유자>/<이름>/
    # 이름으로 판별하려면 실제 데이터셋 폴더까지 내려가야 한다.
    base = pathlib.Path("/kaggle/input")
    out = []
    for first in sorted(base.iterdir()):
        if not first.is_dir():
            continue
        if first.name in ("datasets", "competitions", "models", "notebooks"):
            for owner in sorted(first.iterdir()):
                if owner.is_dir():
                    out += [d for d in sorted(owner.iterdir()) if d.is_dir()]
        else:
            out.append(first)
    return out


def autodetect():
    found = {"voice_real": [], "voice_fake": [], "music_real": [], "music_fake": []}
    # 가짜 음악은 어느 데이터셋에서 왔는지 기억한다. 학습에 쓴 생성기로 검증하면
    # 점수가 뻥튀기되므로 train/holdout 을 생성기 단위로 갈라야 한다.
    fake_music_by_source = {}
    roots = dataset_roots()

    print("[붙어 있는 데이터셋]")
    for r in roots:
        print("  ", r)
    print()

    # ① 생성 음악 데이터셋 — 이름으로 판별한다. real/fake 하위폴더가 없는 게 보통이다.
    for base in roots:
        low = str(base).lower()
        if not any(k in low for k in MUSIC_FAKE_WORDS):
            continue
        real_dirs, fake_dirs = [], []
        for sub in sorted(base.iterdir()) if base.is_dir() else []:
            if not sub.is_dir():
                continue
            (real_dirs if any(k in sub.name.lower() for k in MUSIC_REAL_INSIDE_FAKE)
             else fake_dirs).append(sub)

        picked = []
        for d in (fake_dirs or [base]):
            picked.extend(collect(d, MAX_PER_SOURCE // max(len(fake_dirs), 1)))
        if picked:
            fake_music_by_source[base.name] = picked
            found["music_fake"].extend(picked)
            print(f"  [info] 생성 음악: {base.name} -> {len(picked)}개"
                  + (f" (생성기 {len(fake_dirs)}종)" if fake_dirs else ""))
        if any(k in low for k in VOCAL_RISK_WORDS):
            print(f"  [warn] {base.name} 은 보컬이 섞일 수 있다."
                  " 대회 정의상 음악은 무보컬이므로 PANNs 로 걸러야 한다")

        for d in real_dirs:
            got = collect(d, MAX_PER_SOURCE - len(found["music_real"]))
            found["music_real"].extend(got)
            if got:
                print(f"  [info] 생성셋 안의 진짜 음악: {d.name} -> {len(got)}개")

    # ② 나머지 데이터셋 — real/fake 하위폴더 규칙
    for base in roots:
        low = str(base).lower()
        if any(k in low for k in MUSIC_FAKE_WORDS):
            continue                      # ① 에서 이미 처리했다
        is_music = any(k in low for k in MUSIC_WORDS)
        for sub in base.rglob("*"):
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

    # ③ 진짜 음악 보강 — 음악 데이터셋에는 real/fake 폴더가 없는 게 보통이다.
    # "비어 있을 때만" 찾으면 안 된다. FakeMusicCaps 안의 MusicCaps 가 먼저 채워버리면
    # MUSAN 이 통째로 누락되고, 진짜 음악이 생성셋과 같은 출처 하나로 쏠린다.
    # 출처가 한 곳이면 프로브가 "음악 종류"를 외워버릴 수 있다.
    music_fake_roots = {name for name in fake_music_by_source}
    if len(found["music_real"]) < MAX_PER_SOURCE:
        # 1순위 — music 이라는 하위 폴더. MUSAN 은 music/ noise/ speech/ 로 나뉘므로
        # 통째로 쓰면 사람 말소리가 음악으로 섞인다.
        for base in roots:
            if base.name in music_fake_roots:
                continue
            hits = [d for d in base.rglob("*") if d.is_dir() and d.name.lower() == "music"]
            if hits:
                print("  [info] music 하위 폴더 사용:", hits[0])
                for h in hits:
                    found["music_real"].extend(
                        collect(h, MAX_PER_SOURCE - len(found["music_real"])))
                    if len(found["music_real"]) >= MAX_PER_SOURCE:
                        break
                break
        # 2순위 — 데이터셋 이름이 음악을 가리키면 통째로
        if not found["music_real"]:
            for base in roots:
                low = str(base).lower()
                if base.name in music_fake_roots or "bundle" in low:
                    continue
                if any(k in low for k in MUSIC_WORDS):
                    print("  [info] 이름으로 음악 데이터셋 판별:", base)
                    found["music_real"].extend(collect(base))
                    break

    # 음성 소스에 음악이 섞이지 않게 걸러낸다
    for key in ("voice_real", "voice_fake"):
        found[key] = [q for q in found[key]
                      if not any(k in str(q).lower()
                                 for k in ("/music/", "musan", "gtzan"))]
    found["_fake_music_by_source"] = fake_music_by_source
    return found

SOURCES = autodetect()
# build_valset 은 4개 키만 안다. 출처 기록은 따로 뺀다.
FAKE_MUSIC_BY_SOURCE = SOURCES.pop("_fake_music_by_source", {})
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

# 교차 생성기 분할 — 학습에 쓴 생성기로 검증하면 점수가 뻥튀기된다.
print("\\n[가짜 음악 출처별]")
for name, paths in FAKE_MUSIC_BY_SOURCE.items():
    print(f"  {name:36s} {len(paths):6d}개")
if not FAKE_MUSIC_BY_SOURCE:
    print("  (없음)")
elif len(FAKE_MUSIC_BY_SOURCE) < 2:
    print("  [warn] 생성기가 1종뿐이다. 교차 생성기 검증이 불가능하므로")
    print("         홀드아웃 결과를 일반화 근거로 쓸 수 없다")
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

# 임베딩을 같이 모은다. 프로브 학습에 필요한데, 나중에 켜면 이 50분을 다시 태워야 한다.
# 비용은 메모리뿐이다 — 400파일 기준 100MB 안쪽.
cache = sweep_mod.precompute(script, labels, "/kaggle/working/valset/test",
                             panns, scorer, htdemucs, device, want_embeddings=True)
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
from eval.metric import rank_score

print("최종 순위")
for rank, r in enumerate(results2, 1):
    # Music EER 이 nan 이면 공식 총점도 nan 이다. rank_score 가 측정 가능한 항목만 재정규화한다.
    print(f"{rank:>2}. {r['name']:<28} Score {rank_score(r):.5f}  "
          f"(file {r['file_eer']:.4f} / voice {r['voice_eer']:.4f} / music {r['music_eer']:.4f})")
print("\\n최고 설정 CONFIG 덮어쓰기:")
print(results2[0]["config"])
""")

md("""
## 9. 프로브 학습 — MUSIC 0.27 · FILE 0.45

DF-Arena 는 생성 음악을 학습한 적이 없다. 1B 를 파인튜닝하는 대신 마지막 분류기 직전
임베딩(1280차원)에 선형 하나를 얹는다. **새 의존성 0 · 새 대용량 가중치 0 · 추론 시간 0.**

앞 셀에서 `want_embeddings=True` 로 캐시를 만들었어야 한다. 안 그러면 여기서 멈춘다.

**가짜 음악이 없으면 이 절은 통째로 건너뛴다.** 배울 대상이 없다.
""")
code("""
import numpy as np
from train import fit_probe

MUSIC_HEAD = "/kaggle/working/submit/model/music_head.npz"
FILE_HEAD = "/kaggle/working/submit/model/file_head.npz"

has_fake_music = any(int(r["MUSIC_FAKE_PROB"]) for r in labels)
print("가짜 음악:", "있음" if has_fake_music else "없음 -> 음악 프로브 학습 불가")

music_probe = file_probe = None

if has_fake_music:
    X, y, groups = fit_probe.build_dataset(
        cache, labels, "music", "MUSIC_FAKE_PROB", "MUSIC_PRESENT_PROB")
    cv = fit_probe.cross_validate(X, y, groups)
    probe = fit_probe.fit(X, y)
    print("\\n[음악 프로브]")
    print(fit_probe.describe(probe, X, y, cv))
    fit_probe.save(MUSIC_HEAD, probe)
    music_probe = script.LinearProbe(MUSIC_HEAD)

# FILE 프로브는 가짜 음악이 없어도 학습된다 (음성 위조만으로도 FILE 라벨이 선다).
# 다만 그 경우 "음악이 진짜일 때"만 배운 프로브다.
Xf, yf, gf = fit_probe.build_dataset(cache, labels, "original", "FILE_FAKE_PROB")
cvf = fit_probe.cross_validate(Xf, yf, gf)
probe_f = fit_probe.fit(Xf, yf)
print("\\n[파일 프로브]")
print(fit_probe.describe(probe_f, Xf, yf, cvf))
fit_probe.save(FILE_HEAD, probe_f)
file_probe = script.LinearProbe(FILE_HEAD)

print("\\n교차검증 AUC 가 0.5 근처면 임베딩에 신호가 없다는 뜻이다 — 프로브를 쓰지 않는다.")
""")

md("""
## 10. 프로브 스윕

**`file_head` 재평가가 이 절의 핵심이다.** 제출 #2 에서 `direct` 가 `fusion` 을 이긴 것은
"음악 가지가 쓰레기일 때"의 결론이었다. 음악 프로브가 서면 융합이 다시 이길 수 있고,
그러면 MUSIC(0.27) 뿐 아니라 FILE(0.45)까지 회수된다.
""")
code("""
probe_configs = [(best["name"], best["config"])] + sweep_mod.probe_grid()
results3 = sweep_mod.sweep(script, cache, labels, probe_configs,
                           music_probe=music_probe, file_probe=file_probe,
                           baseline_name=best["name"])

print("\\n최고 설정:", results3[0]["name"], results3[0]["config"])
print("\\n주의 — 이 검증셋으로 고른 프로브는 같은 생성기에 과적합했을 수 있다.")
print("      학습에 쓰지 않은 생성기로 만든 홀드아웃에서 이득이 유지되는지 반드시 확인한다.")
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
