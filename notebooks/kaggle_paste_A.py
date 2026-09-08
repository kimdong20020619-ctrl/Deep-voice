# ===== A 블록 : 1단계 — 준비와 채점 (GPU 필요, 30~90분) =====
# Kaggle 새 노트북에 이 전체를 붙여넣고 실행한다.
# Import Notebook 이 안 될 때 쓰는 우회로이며 내용은 노트북과 동일하다.


# ---------- A1 ----------
import torch, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "Settings -> Accelerator -> GPU 로 바꿔라"

import glob
print("\n[붙어 있는 입력]")
for path in sorted(glob.glob("/kaggle/input/*")):
    print(" ", path)

# ---------- A2 ----------
!pip -q install demucs==4.0.1 panns-inference==0.1.1 2>&1 | tail -5

import importlib, torch
for mod in ("demucs", "panns_inference", "librosa", "soundfile", "torchaudio"):
    try:
        m = importlib.import_module(mod)
        print(f"  OK   {mod:18s} {getattr(m, '__version__', '')}")
    except Exception as error:
        print(f"  FAIL {mod:18s} {error}")

print("\ntorch", torch.__version__, "| cuda", torch.cuda.is_available())
assert torch.cuda.is_available(), "설치 과정에서 torch 가 바뀌었다 — Run > Restart Session 후 처음부터"

# ---------- A3 ----------
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
print("\n[번들 내용]")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print(f"  {p.stat().st_size/1024:8.1f} KB  {p.relative_to(WORK)}")

# ---------- A4 ----------
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
    print("\n[warn] 가짜 음악 소스가 없다 -> Music EER 은 nan 이 된다 (실효 가중치 0.27 미측정)")

# 교차 생성기 분할 — 학습에 쓴 생성기로 검증하면 점수가 뻥튀기된다.
print("\n[가짜 음악 출처별]")
for name, paths in FAKE_MUSIC_BY_SOURCE.items():
    print(f"  {name:36s} {len(paths):6d}개")
if not FAKE_MUSIC_BY_SOURCE:
    print("  (없음)")
elif len(FAKE_MUSIC_BY_SOURCE) < 2:
    print("  [warn] 생성기가 1종뿐이다. 교차 생성기 검증이 불가능하므로")
    print("         홀드아웃 결과를 일반화 근거로 쓸 수 없다")

# ---------- A5 ----------
import importlib, sys
sys.path.insert(0, "/kaggle/working/submit/src")
from synth import build_valset
importlib.reload(build_valset)

VALSET_DIR = "/kaggle/working/valset"
N_CLIPS = 400          # 먼저 400 으로 확인하고, 시간이 되면 늘린다

rows = build_valset.build(SOURCES, VALSET_DIR, count=N_CLIPS, seed=0)

# ---------- A6 ----------
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

expected = {}
for line in (MODEL / "SHA256SUMS.txt").read_text().splitlines():
    if line.strip():
        digest, rel = line.split()
        expected[rel] = digest

snap = snapshot_download(repo_id="Speech-Arena-2025/DF_Arena_1B_V_1", revision="fb6ce85de12c2c5a509d89114adaf827dd75f49f",
                         allow_patterns=["pytorch_model.bin"])
shutil.copy2(pathlib.Path(snap) / "pytorch_model.bin", MODEL / "df_arena_1b" / "pytorch_model.bin")

from demucs.pretrained import get_model as _dl
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
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
    print(f"[{'OK  ' if got == digest else 'FAIL'}] {rel}")
print("\n무결성", "일치 — 대회 배포본과 같은 가중치다" if ok else "불일치! 신뢰할 수 없다")
assert ok

# ---------- A7 ----------
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

# ---------- A8 ----------
# 캐시를 저장해두면 이후 스윕은 GPU 없이도 돌릴 수 있다
import pickle, pathlib
with open("/kaggle/working/cache.pkl", "wb") as f:
    pickle.dump({"cache": cache, "labels": labels}, f)
print("cache.pkl", pathlib.Path("/kaggle/working/cache.pkl").stat().st_size / 1024**2, "MB")
