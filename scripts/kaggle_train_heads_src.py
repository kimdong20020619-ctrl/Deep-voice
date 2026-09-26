# -*- coding: utf-8 -*-
# Kaggle 학습 노트북 원본. scripts/make_kaggle_train_notebook.py 가 "# %%" 단위로 셀을 나눠
# notebooks/kaggle_train_heads.ipynb 를 만든다. SMOKE=1 이면 다운로드·GPU 없이 합성 신호와
# 가짜 채점기로 전체 흐름만 점검한다(로컬 CPU).
#
# 목적: 공개 데이터로 음성/음악/혼합 학습셋을 합성하고, 제출 코드와 같은 DFArenaScorer 로
#       세그먼트 임베딩(1280차원)을 뽑아 FILE·VOICE·MUSIC 선형 헤드를 학습한다.
#       출력 헤드는 submit/script.py 의 LinearProbe(npz: w, b, mean, scale) 형식이다.

# %% [1] 환경과 설정
import base64
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import types
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SMOKE = os.environ.get("SMOKE") == "1"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # 진행률 위젯이 브라우저 탭을 멈추게 했다 (09-25)
SEED = 20260925
random.seed(SEED)
rng = np.random.default_rng(SEED)
SR = 16_000
STARTED = time.time()
TIME_LIMIT = 10.5 * 3600          # Kaggle 12시간 세션 — 학습·저장 시간을 남긴다

KAGGLE = Path("/kaggle/working").exists()
if SMOKE:
    WORK = Path(os.environ.get("SMOKE_DIR", "smoke_out")).resolve()
    TMP = WORK / "tmp"
elif KAGGLE:
    WORK = Path("/kaggle/working")
    TMP = Path("/tmp/dv")
else:                      # Colab
    WORK = Path("/content/out")
    TMP = Path("/content/dv")
OUT = WORK / "dv_heads"
for folder in (OUT, TMP):
    folder.mkdir(parents=True, exist_ok=True)
LOG_FILE = open(OUT / "run.log", "a", encoding="utf-8")


def log(*parts):
    message = f"[{(time.time() - STARTED) / 60:6.1f}m] " + " ".join(str(p) for p in parts)
    print(message, flush=True)
    LOG_FILE.write(message + "\n")
    LOG_FILE.flush()


# 규모 — 합계 약 1.1만 클립. T4 기준 임베딩 추출 1.5시간 내외(0.059 s/오디오초 실측 기준).
COUNTS = dict(
    train=dict(v_real=2000, v_fake=2000, m_real=1200, m_fake=1200, mix_sim=2400, mix_seq=800),
    hold=dict(v_real=300, v_fake=300, m_real=200, m_fake=200, mix_sim=320, mix_seq=80),
)
if SMOKE:
    COUNTS = dict(train=dict(v_real=12, v_fake=12, m_real=8, m_fake=8, mix_sim=16, mix_seq=8),
                  hold=dict(v_real=6, v_fake=6, m_real=4, m_fake=4, mix_sim=8, mix_seq=4))

# 생성기 단위 홀드아웃 — 학습에서 본 적 없는 생성기로만 일반화를 판단한다.
HOLD_VOICE_MODELS = {"minimax_speech-02-turbo", "VoxCPM2"}
HOLD_MUSIC_MODELS = {"stable_audio_open"}      # 대소문자 무시 비교
DF_REPO = "Speech-Arena-2025/DF_Arena_1B_V_1"
DF_REV = "fb6ce85de12c2c5a509d89114adaf827dd75f49f"
DF_SHA = "780bc14fd4c15e65d58efdef728427cf03cd29cd60be528e97badf8c89087988"
FMC_URL = "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
ZEROTH_REPO = "Bingsu/zeroth-korean"

HF_TOKEN = os.environ.get("HF_TOKEN")
if not SMOKE:
    if KAGGLE and not HF_TOKEN:
        try:
            from kaggle_secrets import UserSecretsClient
            HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as error:
            log("[warn] HF_TOKEN secret 없음:", type(error).__name__)
    import torch
    assert torch.cuda.is_available(), "GPU 가 켜져 있지 않다 — Settings > Accelerator"
    log("torch", torch.__version__, torch.cuda.get_device_name(0))

# MLAAD(게이트) 토큰이 없으면 음악 헤드만 학습한다 — 음악 데이터는 로그인 없이 받을 수 있다.
MODE = "full" if (HF_TOKEN or os.environ.get("SMOKE_MODE") == "full") else "music"
if MODE == "music":
    COUNTS = {split: dict(c, v_real=0, v_fake=0) for split, c in COUNTS.items()}
log("MODE", MODE, "KAGGLE" if KAGGLE else "COLAB/LOCAL")

# %% [2] 제출 코드와 DF-Arena 준비 — 추론과 똑같은 전처리·임베딩을 쓰기 위해 script.py 를 그대로 쓴다
EMBEDDED = json.loads(base64.b64decode("__EMBEDDED_FILES__").decode("utf-8")) if not SMOKE else {}
SUB = TMP / "submit"
for relative, blob in EMBEDDED.items():
    path = SUB / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(blob))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


if SMOKE:
    class FakeScorer:
        """로컬 점검용 — 오디오 통계로 결정적인 가짜 임베딩을 만든다."""
        has_embeddings = True

        def score(self, audio, kind=None, want_embeddings=False):
            n = max(1, int(math.ceil(audio.size / 64_600)))
            base = np.array([audio.std(), np.abs(audio).mean()], dtype=np.float32)
            emb = np.tile(np.resize(base, 1280), (n, 1)) + rng.normal(0, 0.01, (n, 1280)).astype(np.float32)
            return (float(np.tanh(audio.std() * 10)), emb) if want_embeddings else 0.5

    scorer = FakeScorer()

    def separate(audio):
        return audio * 0.5, audio * 0.5
else:
    try:
        import demucs  # noqa: F401
    except ImportError:
        # 서버와 같은 버전. torch 는 건드리지 않도록 의존성 목록만 확인하고 설치한다.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "demucs==4.0.1"], check=True)
    import torch
    HT_SHA = "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
    ht_dir = SUB / "model" / "htdemucs"
    ht_dir.mkdir(parents=True, exist_ok=True)
    ht_weights = ht_dir / "955717e8-8726e21a.th"
    if not ht_weights.exists():
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th", ht_weights)
    assert sha256_file(ht_weights) == HT_SHA, "HTDemucs 가중치 해시 불일치"
    log("HTDemucs 가중치 해시 일치")
    from huggingface_hub import hf_hub_download
    weights = SUB / "model" / "df_arena_1b" / "pytorch_model.bin"
    cached = hf_hub_download(DF_REPO, "pytorch_model.bin", revision=DF_REV)
    shutil.copyfile(cached, weights)
    assert sha256_file(weights) == DF_SHA, "DF-Arena 가중치 해시 불일치"
    log("DF-Arena 가중치 해시 일치")
    spec = importlib.util.spec_from_file_location("dv_submit", SUB / "script.py")
    dv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dv)
    dv.CONFIG["segment_agg"] = "topk_mean"       # 현재 최고점 제출(topk25)과 같은 집계
    dv.CONFIG["segment_topk_ratio"] = 0.25
    scorer = dv.DFArenaScorer(torch.device("cuda"))
    assert scorer.has_embeddings, "fc5 임베딩 훅이 설치되지 않았다"
    probe = (np.random.default_rng(0).standard_normal(SR * 6) * 0.05).astype(np.float32)
    _, probe_emb = scorer.score(probe, want_embeddings=True)
    assert probe_emb is not None and probe_emb.shape[1] == 1280, probe_emb
    log("DFArenaScorer 준비 완료, 임베딩", probe_emb.shape)
    # 혼합 파일의 MUSIC 은 추론에서 반주 스템으로 채점된다 — 학습도 같은 분리를 거친다.
    htdemucs = dv.load_htdemucs_model(torch.device("cuda"))

    def separate(audio):
        return dv.separate_voice_and_music(audio, htdemucs, torch.device("cuda"))

    _, stem_check = separate(probe)
    log("HTDemucs 분리 확인", stem_check.shape)

# %% [3] 오디오 유틸 — 레벨·증강·혼합
import librosa
from scipy.signal import butter, resample_poly, sosfilt

PHONE_SOS = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def set_level(audio, dbfs):
    return (audio * (10 ** (dbfs / 20) / max(rms(audio), 1e-6))).astype(np.float32)


def peak_safe(audio):
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return (audio * (0.97 / peak)).astype(np.float32) if peak > 0.97 else audio.astype(np.float32)


def crop(audio, seconds, rnd):
    length = int(seconds * SR)
    if audio.size <= length:
        return audio.astype(np.float32)
    start = int(rnd.integers(0, audio.size - length + 1))
    return audio[start:start + length].astype(np.float32)


def aug_mp3(audio, bitrate):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
         "-codec:a", "libmp3lame", "-b:a", f"{bitrate}k", "-f", "mp3", "pipe:1"],
        input=audio.astype("<f4").tobytes(), capture_output=True, check=True).stdout
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-f", "mp3", "-i", "pipe:0", "-f", "f32le", "-ar", str(SR),
         "-ac", "1", "pipe:1"], input=raw, capture_output=True, check=True).stdout
    out = np.frombuffer(decoded, dtype="<f4").copy()
    return out[:audio.size] if out.size >= audio.size else np.pad(out, (0, audio.size - out.size))


def aug_phone(audio):
    """전화채널 근사: 8kHz 대역 제한 + μ-law 8비트 양자화."""
    narrow = sosfilt(PHONE_SOS, resample_poly(audio, 1, 2))
    peak = max(float(np.max(np.abs(narrow))), 1e-6)
    x = np.clip(narrow / peak, -1, 1)
    mu = 255.0
    q = np.round((np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu) + 1) / 2 * mu) / mu * 2 - 1
    x = np.sign(q) * (np.power(1 + mu, np.abs(q)) - 1) / mu * peak
    return resample_poly(x, 2, 1)[:audio.size].astype(np.float32)


NOISES = []   # MUSAN noise — [4] 에서 채운다


def augment(audio, rnd):
    """REAL·FAKE 에 같은 확률로 건다. 후처리 자체가 FAKE 단서가 되지 않게 한다."""
    applied = []
    if NOISES and rnd.random() < 0.3:
        noise = NOISES[int(rnd.integers(len(NOISES)))]
        noise = np.resize(crop(noise, audio.size / SR, rnd), audio.size)
        snr = float(rnd.uniform(5, 30))
        audio = audio + noise * (rms(audio) / max(rms(noise), 1e-6)) * 10 ** (-snr / 20)
        applied.append(f"noise{snr:.0f}")
    if rnd.random() < 0.2:
        audio = aug_phone(audio)
        applied.append("phone")
    elif rnd.random() < 0.35:
        bitrate = int(rnd.choice([32, 48, 64, 96, 128]))
        audio = aug_mp3(audio, bitrate)
        applied.append(f"mp3_{bitrate}")
    audio = set_level(audio, float(rnd.uniform(-32, -14)))
    return peak_safe(audio), applied


# %% [4] 데이터 수집 — 각 출처는 실패해도 로그를 남기고, 필수 출처가 비면 중단한다
SOURCES = defaultdict(list)   # key: v_real / v_fake / m_real / m_fake / m_song(보컬 있는 진짜 곡)


def add_source(key, audio, **meta):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 4 * SR or not np.isfinite(audio).all() or rms(audio) < 1e-4:
        return
    # 메모리 절약: 최대 12초만 int16 로 보관
    if audio.size > 12 * SR:
        start = (audio.size - 12 * SR) // 2
        audio = audio[start:start + 12 * SR]
    scale = max(float(np.max(np.abs(audio))), 1e-6)
    SOURCES[key].append(dict(meta, key=key, pcm=(audio / scale * 32000).astype(np.int16)))


def pcm_float(source):
    return source["pcm"].astype(np.float32) / 32000


def synth_smoke_sources():
    """SMOKE 전용: 톤·잡음으로 네 종류 소스를 흉내 낸다."""
    t = np.arange(8 * SR) / SR
    for i in range(40):
        add_source("v_real", np.sin(2 * np.pi * (120 + i) * t) * (1 + np.sin(3 * t)), corpus="smoke", group=f"spk{i % 8}", split="hold" if i % 8 == 0 else "train")
        add_source("v_fake", np.sign(np.sin(2 * np.pi * (140 + i) * t)) * 0.5, corpus="smoke", group="HoldTTS" if i % 5 == 0 else f"tts{i % 3}", split="hold" if i % 5 == 0 else "train")
        add_source("m_real", rng.standard_normal(t.size) * 0.1 + np.sin(2 * np.pi * 440 * t), corpus="smoke", group=f"alb{i % 6}", split="hold" if i % 6 == 0 else "train")
        add_source("m_fake", np.sin(2 * np.pi * 660 * t) ** 3, corpus="smoke", group="hold_gen" if i % 5 == 0 else f"gen{i % 3}", split="hold" if i % 5 == 0 else "train")
    for i in range(20):
        NOISES.append(rng.standard_normal(4 * SR).astype(np.float32) * 0.1)


def collect_mlaad():
    from huggingface_hub import snapshot_download
    assert HF_TOKEN, "HF_TOKEN 이 필요하다 — MLAAD 는 게이트 데이터셋이다"
    root = Path(snapshot_download("mueller91/MLAAD", repo_type="dataset", allow_patterns=["fake/ko/**"],
                                  token=HF_TOKEN, local_dir=str(TMP / "mlaad"), max_workers=16))
    for model_dir in sorted((root / "fake" / "ko").iterdir()):
        if not model_dir.is_dir():
            continue
        files = sorted(model_dir.glob("*.wav"))
        split = "hold" if model_dir.name in HOLD_VOICE_MODELS else "train"
        keep = files[:260] if split == "train" else files[:200]
        for path in keep:
            try:
                audio, _ = librosa.load(path, sr=SR, mono=True)
                add_source("v_fake", audio, corpus="MLAAD-ko", group=model_dir.name, split=split, file=path.name)
            except Exception as error:
                log("[warn] MLAAD", path.name, type(error).__name__)
        log("MLAAD", model_dir.name, split, len(keep))


def collect_zeroth():
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download, list_repo_files
    files = [f for f in list_repo_files(ZEROTH_REPO, repo_type="dataset") if f.endswith(".parquet")]
    chosen = [f for f in files if "/train-0000" in f][:2] + [f for f in files if "/test-" in f][:1]
    for name in chosen:
        split = "hold" if "/test-" in name else "train"
        table = pq.read_table(hf_hub_download(ZEROTH_REPO, name, repo_type="dataset"))
        columns = table.column_names
        audio_col = next(c for c in columns if c.lower() == "audio")
        speaker_col = next((c for c in columns if "speaker" in c.lower()), None)
        rows = table.to_pylist()
        rnd = np.random.default_rng(len(rows))
        order = rnd.permutation(len(rows))[: (2600 if split == "train" else 420)]
        for idx in order:
            row = rows[int(idx)]
            cell = row[audio_col]
            try:
                blob = cell["bytes"] if isinstance(cell, dict) else cell
                audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != SR:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
                add_source("v_real", audio, corpus="Zeroth-Korean", split=split,
                           group=f"spk{row.get(speaker_col)}" if speaker_col else name)
            except Exception as error:
                log("[warn] Zeroth", type(error).__name__)
        log("Zeroth", name, split, len(order), "columns", columns)


class RangeFile(io.RawIOBase):
    """HTTP Range 로 읽는 원격 파일 — FakeMusicCaps 12.9GB zip 에서 필요한 멤버만 받는다."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, index):
        if index not in self.cache:
            start = index * self.block
            end = min(self.size, start + self.block) - 1
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request, timeout=60) as r:
                        self.cache[index] = r.read()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[index]

    def readinto(self, buffer):
        if self.pos >= self.size:
            return 0
        index, offset = divmod(self.pos, self.block)
        data = self._block(index)[offset:offset + len(buffer)]
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch(url, local):
    """큰 파일을 병렬 연결로 받는다. Zenodo 는 연결당 약 2MB/s 로 제한됐다(09-26 실측, 단일 curl)."""
    began = time.time()
    if shutil.which("aria2c") is None:
        subprocess.run(["apt-get", "-qq", "install", "-y", "aria2"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if shutil.which("aria2c"):
        subprocess.run(["aria2c", "-q", "-x16", "-s16", "-k4M", "--file-allocation=none", "--max-tries=10",
                        "--retry-wait=3", "-d", str(local.parent), "-o", local.name, url], check=True)
    else:
        subprocess.run(["curl", "-sS", "-L", "--retry", "5", "-C", "-", "-o", str(local), url], check=True)
    log(f"다운로드 {local.name} {local.stat().st_size / 2**30:.1f} GB, {(time.time() - began) / 60:.1f}분")


def collect_fakemusiccaps():
    # 파일마다 Range 요청을 보내면 Zenodo 지연 때문에 1,900개에 90분 이상 걸린다(09-26 실측).
    # 12.9GB 를 병렬로 한 번에 받고 로컬에서 읽는다. 실패하면 Range 방식으로 물러선다.
    local = TMP / "FakeMusicCaps.zip"
    try:
        fetch(FMC_URL, local)
        archive = zipfile.ZipFile(local)
    except Exception as error:
        log("[warn] 전체 다운로드 실패, Range 방식으로 전환:", type(error).__name__, error)
        archive = zipfile.ZipFile(io.BufferedReader(RangeFile(FMC_URL), buffer_size=1 << 20))
    by_generator = defaultdict(list)
    for name in archive.namelist():
        # zip 에 macOS 리소스 포크(__MACOSX/._*.wav)가 섞여 있다 — 09-26 첫 실행에서 절반이 읽기 실패했다
        if name.lower().endswith(".wav") and "__MACOSX" not in name and not name.split("/")[-1].startswith("._"):
            parts = name.split("/")
            by_generator[parts[-2]].append(name)
    log("FakeMusicCaps 생성기", {g: len(v) for g, v in by_generator.items()})
    for generator, names in sorted(by_generator.items()):
        split = "hold" if generator.lower() in HOLD_MUSIC_MODELS else "train"
        pick = np.random.default_rng(len(names)).permutation(len(names))[: (380 if split == "train" else 260)]
        for idx in pick:
            name = names[int(idx)]
            try:
                audio, _ = librosa.load(io.BytesIO(archive.read(name)), sr=SR, mono=True)
                add_source("m_fake", audio, corpus="FakeMusicCaps", group=generator, split=split, file=name)
            except Exception as error:
                log("[warn] FMC", name, type(error).__name__)
        log("FakeMusicCaps", generator, split, len(pick))


def collect_musan():
    """music(보컬 여부 주석 포함)과 noise 만 풀고 speech 는 건너뛴다."""
    root = TMP / "musan"

    class Counting(io.RawIOBase):
        """11GB 스트리밍 진행을 1GB 마다 로그로 남긴다 (진행률 표시줄은 탭을 멈추게 했다)."""

        def __init__(self, raw):
            self.raw, self.total, self.mark = raw, 0, 0

        def readable(self):
            return True

        def readinto(self, buffer):
            n = self.raw.readinto(buffer)
            self.total += n or 0
            if self.total - self.mark >= 1 << 30:
                self.mark = self.total
                log(f"MUSAN 수신 {self.total / 2**30:.0f} GB")
            return n

    tar_path = TMP / "musan.tar.gz"
    try:
        fetch(MUSAN_URL, tar_path)
        opened = tarfile.open(tar_path, mode="r|gz")
    except Exception as error:
        log("[warn] MUSAN 병렬 다운로드 실패, 스트리밍으로 전환:", type(error).__name__, error)
        opened = tarfile.open(fileobj=io.BufferedReader(Counting(urllib.request.urlopen(MUSAN_URL, timeout=120)), 1 << 20),
                              mode="r|gz")
    with opened as tar:
        for member in tar:
            if member.isfile() and (member.name.startswith("musan/music/") or member.name.startswith("musan/noise/")):
                tar.extract(member, root)
    vocals = {}
    for ann in (root / "musan" / "music").glob("*/ANNOTATIONS"):
        for line in ann.read_text(encoding="utf-8", errors="ignore").splitlines():
            fields = line.split()
            if len(fields) >= 3:
                vocals[fields[0]] = fields[2].upper().startswith("Y")
    music_files = sorted((root / "musan" / "music").rglob("*.wav"))
    for index, path in enumerate(music_files):
        split = "hold" if int(hashlib.md5(path.stem.encode()).hexdigest(), 16) % 7 == 0 else "train"
        try:
            audio, _ = librosa.load(path, sr=SR, mono=True, duration=40)
        except Exception as error:
            log("[warn] MUSAN", path.name, type(error).__name__)
            continue
        key = "m_song" if vocals.get(path.stem, False) else "m_real"
        for k in range(3):        # 곡당 서로 다른 구간 3개
            piece = audio[k * len(audio) // 3:(k + 1) * len(audio) // 3]
            add_source(key, piece, corpus="MUSAN-music", group=path.parent.name + "/" + path.stem, split=split)
    for path in sorted((root / "musan" / "noise").rglob("*.wav"))[:400]:
        try:
            noise, _ = librosa.load(path, sr=SR, mono=True, duration=20)
            if noise.size >= SR:
                NOISES.append(noise.astype(np.float32))
        except Exception:
            pass
    log("MUSAN music", len(music_files), "vocals 주석", sum(vocals.values()), "noise", len(NOISES))


if SMOKE:
    synth_smoke_sources()
else:
    plan = [("Zeroth", collect_zeroth, True), ("FakeMusicCaps", collect_fakemusiccaps, True),
            ("MUSAN", collect_musan, True)]
    if MODE == "full":
        plan.insert(0, ("MLAAD", collect_mlaad, True))
    for name, collector, required in plan:
        began = time.time()
        try:
            collector()
        except Exception:
            log(f"[error] {name} 수집 실패\n{traceback.format_exc()}")
            if required:
                raise
        log(f"{name} 완료 {(time.time() - began) / 60:.1f}분")

summary = {key: Counter(s["split"] for s in items) for key, items in SOURCES.items()}
log("소스 수", json.dumps({k: dict(v) for k, v in summary.items()}, ensure_ascii=False))
for key in (("v_real", "v_fake", "m_real", "m_fake") if MODE == "full" else ("v_real", "m_real", "m_fake")):
    for split in ("train", "hold"):
        assert summary.get(key, {}).get(split, 0) > 0, f"{key}/{split} 소스가 비었다"

# %% [5] 레시피 생성 — 성분 라벨은 대회 정의를 따른다 (하나라도 FAKE 면 FILE=1)
def pick(key, split, rnd):
    pool = [s for s in SOURCES[key] if s["split"] == split]
    return pool[int(rnd.integers(len(pool)))]


def make_recipes(split, counts, rnd):
    recipes = []

    def voice_source(fake):
        return pick("v_fake" if fake else "v_real", split, rnd)

    has_song = any(s["split"] == split for s in SOURCES.get("m_song", []))

    def music_source(fake):
        # 보컬 있는 진짜 곡(MUSAN vocals=Y)도 일부 섞는다 — 평가셋의 '노래 = 혼합' 을 흉내 낸다
        if not fake and has_song and rnd.random() < 0.15:
            return pick("m_song", split, rnd)
        return pick("m_fake" if fake else "m_real", split, rnd)

    for _ in range(counts["v_real"]):
        recipes.append(dict(mode="voice", voice=voice_source(0), vf=0))
    for _ in range(counts["v_fake"]):
        recipes.append(dict(mode="voice", voice=voice_source(1), vf=1))
    for _ in range(counts["m_real"]):
        recipes.append(dict(mode="music", music=music_source(0), mf=0))
    for _ in range(counts["m_fake"]):
        recipes.append(dict(mode="music", music=music_source(1), mf=1))
    combos = [(0, 0), (1, 0), (0, 1), (1, 1)] if MODE == "full" else [(0, 0), (0, 1)]
    for mode, total in (("mix_sim", counts["mix_sim"]), ("mix_seq", counts["mix_seq"])):
        for i in range(total):
            vf, mf = combos[i % len(combos)]
            recipes.append(dict(mode=mode, voice=voice_source(vf), vf=vf, music=music_source(mf), mf=mf))
    for r in recipes:
        r["split"] = split
        r["seconds"] = float(rnd.uniform(4.1, 8.0))
        r["music_rel_db"] = float(rnd.uniform(-20, 0))
        # 보컬 있는 진짜 곡은 음성 성분도 가진다 (보컬 = 음성)
        song = r.get("music", {}).get("key") == "m_song"
        voice_present = int(r["mode"] != "music" or song)
        music_present = int(r["mode"] != "voice")
        vf = r.get("vf", 0)
        mf = r.get("mf", 0)
        r["labels"] = dict(FILE=int(vf or mf), VOICE_PRESENT=voice_present, MUSIC_PRESENT=music_present,
                           VOICE_FAKE=vf if voice_present else None, MUSIC_FAKE=mf if music_present else None)
    return recipes


def render(recipe, rnd):
    seconds = recipe["seconds"]
    if recipe["mode"] == "voice":
        audio = crop(pcm_float(recipe["voice"]), seconds, rnd)
    elif recipe["mode"] == "music":
        audio = crop(pcm_float(recipe["music"]), seconds, rnd)
    elif recipe["mode"] == "mix_sim":
        voice = set_level(crop(pcm_float(recipe["voice"]), seconds, rnd), -20)
        music = set_level(np.resize(crop(pcm_float(recipe["music"]), seconds, rnd), voice.size), -20 + recipe["music_rel_db"])
        audio = voice + music
    else:   # mix_seq — 음성과 음악이 순차로 이어진다 (대회 정의상 혼합)
        half = seconds / 2
        voice = set_level(crop(pcm_float(recipe["voice"]), half, rnd), -20)
        music = set_level(crop(pcm_float(recipe["music"]), half, rnd), -20 + recipe["music_rel_db"] / 2)
        parts = [voice, music] if rnd.random() < 0.5 else [music, voice]
        fade = int(0.05 * SR)
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        parts[0][-fade:] *= ramp[::-1]
        parts[1][:fade] *= ramp
        audio = np.concatenate(parts)
    if audio.size < int(4.04 * SR):   # 평가셋 최소 4초 — 짧은 소스는 반복으로 채운다
        audio = np.resize(audio, int(4.04 * SR))
    return augment(audio.astype(np.float32), rnd)


rnd_train = np.random.default_rng(SEED + 1)
rnd_hold = np.random.default_rng(SEED + 2)
RECIPES = make_recipes("train", COUNTS["train"], rnd_train) + make_recipes("hold", COUNTS["hold"], rnd_hold)
log("레시피", len(RECIPES), Counter((r["split"], r["mode"]) for r in RECIPES))

# %% [6] 임베딩 추출 — 제출 코드의 DFArenaScorer.score(want_embeddings=True) 그대로
# 두 가지 시점을 저장한다.
#   X  : 원본 오디오 임베딩 — FILE(file_probe·direct)·VOICE(direct) 헤드용
#   XM : 음악 시점 — 음성·음악이 함께 있으면 추론처럼 HTDemucs 반주 스템, 음악만 있으면 원본.
#        제출 코드의 music_head=probe 가 적용되는 입력과 같다 (게이팅 임계값 대신 정답 존재 라벨로 근사).
emb_chunks, seg_owner, music_chunks, music_owner, clip_rows = [], [], [], [], []
rnd_render = np.random.default_rng(SEED + 3)
for index, recipe in enumerate(RECIPES):
    if time.time() - STARTED > TIME_LIMIT:
        log("[warn] 시간 한도 — 여기까지만 쓴다", index)
        break
    labels_ = recipe["labels"]
    try:
        audio, applied = render(recipe, rnd_render)
        base_score, emb = scorer.score(audio, want_embeddings=True)
        music_score, music_emb = None, None
        if labels_["MUSIC_PRESENT"]:
            if labels_["VOICE_PRESENT"]:
                _, stem = separate(audio)
                music_score, music_emb = scorer.score(stem, want_embeddings=True)
            else:
                music_score, music_emb = base_score, emb
    except Exception as error:
        log("[warn] 렌더/채점 실패", index, type(error).__name__, error)
        continue
    if emb is None or emb.shape[0] == 0:
        continue
    row = dict(index=index, split=recipe["split"], mode=recipe["mode"], seconds=round(audio.size / SR, 3),
               aug="+".join(applied), df_score=float(base_score),
               df_music_score=None if music_score is None else float(music_score),
               voice_group=recipe.get("voice", {}).get("group"), voice_corpus=recipe.get("voice", {}).get("corpus"),
               music_group=recipe.get("music", {}).get("group"), music_corpus=recipe.get("music", {}).get("corpus"),
               **{k: v for k, v in labels_.items()})
    clip_rows.append(row)
    clip_id = len(clip_rows) - 1
    emb_chunks.append(emb.astype(np.float16))
    seg_owner.extend([clip_id] * emb.shape[0])
    if music_emb is not None and music_emb.shape[0] > 0:
        music_chunks.append(music_emb.astype(np.float16))
        music_owner.extend([clip_id] * music_emb.shape[0])
    if (index + 1) % 250 == 0:
        log(f"임베딩 {index + 1}/{len(RECIPES)}")

X = np.concatenate(emb_chunks).astype(np.float32)
OWNER = np.asarray(seg_owner)
XM = np.concatenate(music_chunks).astype(np.float32) if music_chunks else np.zeros((0, 1280), np.float32)
OWNER_M = np.asarray(music_owner, dtype=int)
log("세그먼트 원본", X.shape, "음악 시점", XM.shape, "클립", len(clip_rows))

# %% [7] 헤드 학습과 평가 — 세그먼트 로지스틱 회귀, 클립은 topk25 로 집계 (제출과 동일)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


def eer(labels, scores):
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def topk25(values):
    values = np.sort(np.asarray(values))
    k = max(1, int(math.ceil(values.size * 0.25)))
    return float(values[-k:].mean())


# head: (라벨 열, 특징, 세그먼트 소유 클립, 비교 기준 = 현재 제출이 그 축에 쓰는 DF-Arena 점수)
HEADS = {"file": ("FILE", X, OWNER, "df_score"),
         "voice": ("VOICE_FAKE", X, OWNER, "df_score"),
         "music": ("MUSIC_FAKE", XM, OWNER_M, "df_music_score")}
if MODE == "music":
    HEADS = {"music": HEADS["music"]}   # 가짜 음성 없이 FILE·VOICE 헤드를 배우면 음악 위조만 FILE 로 배운다
metrics = {}
clip_split = np.array([r["split"] for r in clip_rows])

for head, (column, FEAT, OWN, base_key) in HEADS.items():
    labels = np.array([np.nan if r[column] is None else r[column] for r in clip_rows], dtype=float)
    usable = ~np.isnan(labels) & np.isin(np.arange(len(clip_rows)), OWN)
    train_clips = np.where(usable & (clip_split == "train"))[0]
    hold_clips = np.where(usable & (clip_split == "hold"))[0]
    # C 선택용 검증: 학습 클립의 15% (홀드아웃은 선택에 쓰지 않는다)
    perm = np.random.default_rng(SEED + 7).permutation(train_clips)
    val_clips, fit_clips = perm[: len(perm) * 15 // 100], perm[len(perm) * 15 // 100:]

    def segments_of(clips):
        mask = np.isin(OWN, clips)
        return FEAT[mask], labels[OWN[mask]].astype(int), OWN[mask]

    def evaluate(model, scaler, clips):
        xs, _, owner = segments_of(clips)
        probs = model.predict_proba(scaler.transform(xs))[:, 1]
        per_clip = defaultdict(list)
        for p, c in zip(probs, owner):
            per_clip[c].append(p)
        ids = sorted(per_clip)
        return eer(labels[ids], [topk25(per_clip[c]) for c in ids])

    best = None
    for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3):
        xs, ys, _ = segments_of(fit_clips)
        scaler = StandardScaler().fit(xs)
        model = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(scaler.transform(xs), ys)
        val = evaluate(model, scaler, val_clips)
        log(f"{head} C={C} 검증 EER {val:.4f}")
        if best is None or val < best[0]:
            best = (val, C)
    xs, ys, _ = segments_of(train_clips)
    scaler = StandardScaler().fit(xs)
    model = LogisticRegression(C=best[1], class_weight="balanced", max_iter=3000).fit(scaler.transform(xs), ys)
    hold_eer = evaluate(model, scaler, hold_clips)
    df_hold = eer(labels[hold_clips], [clip_rows[c][base_key] for c in hold_clips])
    by_mode = {}
    for mode in sorted({clip_rows[c]["mode"] for c in hold_clips}):
        sub = np.array([c for c in hold_clips if clip_rows[c]["mode"] == mode])
        by_mode[mode] = dict(head=evaluate(model, scaler, sub),
                             df_arena=eer(labels[sub], [clip_rows[c][base_key] for c in sub]), n=int(sub.size))
    # 평균 혼합(blend 0.5)도 같은 홀드아웃에서 본다 — 제출 CONFIG music_head_blend·file_probe_blend 후보
    xs_h, _, own_h = segments_of(hold_clips)
    probs_h = model.predict_proba(scaler.transform(xs_h))[:, 1]
    per_clip = defaultdict(list)
    for p, c in zip(probs_h, own_h):
        per_clip[c].append(p)
    blend_scores = [0.5 * topk25(per_clip[c]) + 0.5 * clip_rows[c][base_key] for c in hold_clips]
    blend_eer = eer(labels[hold_clips], blend_scores)
    metrics[head] = dict(C=best[1], val_eer=best[0], hold_eer=hold_eer, df_arena_hold_eer=df_hold,
                         blend05_hold_eer=blend_eer,
                         n_train_clips=int(train_clips.size), n_hold_clips=int(hold_clips.size), by_mode=by_mode)
    log(f"== {head}: 홀드아웃 EER 헤드 {hold_eer:.4f} · 0.5혼합 {blend_eer:.4f} vs DF-Arena {df_hold:.4f}",
        json.dumps(by_mode))
    np.savez(OUT / f"head_{head}.npz", w=(model.coef_[0]).astype(np.float64), b=float(model.intercept_[0]),
             mean=scaler.mean_.astype(np.float64), scale=scaler.scale_.astype(np.float64))

# 제출 코드의 파일명 규약: file_probe 는 file_head.npz, music_head=probe 는 music_head.npz
for head, alias in (("file", "file_head.npz"), ("music", "music_head.npz")):
    if (OUT / f"head_{head}.npz").exists():
        shutil.copyfile(OUT / f"head_{head}.npz", OUT / alias)

# %% [8] 저장 — 재학습은 로컬 CPU 에서도 할 수 있게 임베딩까지 남긴다
np.savez_compressed(OUT / "embeddings.npz", X=X.astype(np.float16), owner=OWNER,
                    XM=XM.astype(np.float16), owner_m=OWNER_M)
with open(OUT / "clips.csv", "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(clip_rows[0].keys()))
    writer.writeheader()
    writer.writerows(clip_rows)
provenance = dict(
    seed=SEED, counts=COUNTS, hold_voice_models=sorted(HOLD_VOICE_MODELS), hold_music_models=sorted(HOLD_MUSIC_MODELS),
    sources={k: {f"{c}|{g}|{sp}": n for (c, g, sp), n in Counter((s["corpus"], s["group"], s["split"]) for s in v).items()}
             for k, v in SOURCES.items()},
    licenses={"MLAAD-ko": "CC BY-NC 4.0 (HF mueller91/MLAAD, gated)", "Zeroth-Korean": "CC BY 4.0 (HF Bingsu/zeroth-korean)",
              "FakeMusicCaps": "CC BY-NC 4.0 (Zenodo 10.5281/zenodo.15063698)", "MUSAN": "openslr.org/17 (per-file licenses)"},
    df_arena=dict(repo=DF_REPO, revision=DF_REV, sha256=DF_SHA), metrics=metrics,
    minutes=(time.time() - STARTED) / 60, smoke=SMOKE, mode=MODE, platform="kaggle" if KAGGLE else "colab")
(OUT / "summary.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
with zipfile.ZipFile(WORK / "dv_heads_result.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(OUT))
log("완료 — dv_heads_result.zip", json.dumps(metrics, ensure_ascii=False))

# %% [9] 회수용 출력 — 헤드 npz(수십 KB)를 base64 로 찍어 두면 파일 다운로드 없이 옮길 수 있다
for path in sorted(OUT.glob("*.npz")):
    if path.name == "embeddings.npz":
        continue
    blob = path.read_bytes()
    print(f"DVHEAD {path.name} {hashlib.sha256(blob).hexdigest()} {base64.b64encode(blob).decode('ascii')}")
print("DVMETRICS " + json.dumps(metrics, ensure_ascii=False))
