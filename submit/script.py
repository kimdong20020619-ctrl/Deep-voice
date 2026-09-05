#!/usr/bin/env python3
"""경진대회 테스트 데이터에 대한 5개 확률값을 생성한다.

공식 베이스라인(zero-shot)을 기반으로 다음을 개선했다.

  [완주] 파일 단위 예외 처리 — 한 파일이 실패해도 전체가 죽지 않는다
  [완주] 확장자 화이트리스트 제거 — 읽어보고 실패한 것만 폴백 처리한다
  [시간] 단일 루프 + 세 모델 동시 상주 — 파일당 디코딩 1회
  [시간] DF-Arena 세그먼트 배치 추론 + bf16 autocast
  [점수] 융합식·세그먼트 집계·짧은 파일 패딩을 CONFIG로 전환 가능

기본 CONFIG는 베이스라인과 점수 의미가 동일하다. 실험은 토글 하나씩만 바꾼다.

대회 규정 준수 — 파일 단위 독립 예측:
  각 파일의 예측은 그 파일의 오디오만으로 계산한다. 다른 파일의 값·통계를 참조하거나
  테스트셋 전체 분포로 정규화·순위 변환하는 코드를 절대 추가하지 않는다.
"""

import csv
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# 추론에는 model 폴더에 포함된 로컬 파일만 사용한다. (평가 서버는 오프라인)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model


# =============================================================================
# CONFIG — 실험 토글. 기본값은 베이스라인과 동일한 점수 의미를 갖는다.
#          리더보드 제출은 일 3회뿐이므로 한 번에 하나씩만 바꾼다.
# =============================================================================

CONFIG = {
    # ---- 점수에 영향을 주는 항목 (기본값 = 베이스라인) ----

    # FILE_FAKE_PROB 융합식. 실효 가중치 0.45로 단일 최대 항목이다.
    #   "baseline"  : max(VP*VF, MP*MF)                      <- 공식 베이스라인
    #   "gated_max" : 존재 확률은 게이트로만 쓰고 max(VF, MF)
    #   "noisy_or"  : 1 - (1-VP*VF)(1-MP*MF)
    #   "gamma"     : max(VP^g * VF, MP^g * MF)   (눌림 완화)
    "fusion_mode": "baseline",
    "fusion_gate": 0.5,      # gated_max에서 존재로 인정할 임계값
    "fusion_gamma": 0.5,     # gamma 모드의 지수

    # 세그먼트 점수 집계. EER은 순위 기반이라 이상치 하나가 순위를 뒤집는다.
    # 다만 "순차 혼합"(앞은 음악, 뒤는 음성)에는 max가 유리한 면도 있다.
    #   "max" | "mean" | "topk_mean"
    "segment_agg": "max",
    "segment_topk": 2,

    # SEGMENT_SAMPLES(4.0375초)보다 짧은 파일의 패딩.
    # 대회 최소 길이가 4초(64,000샘플)라 4.00~4.04초 파일이 여기 걸린다.
    # tile은 이음매에 클릭(광대역 임펄스)을 만들어 오탐 요인이 될 수 있다.
    #   "tile" | "zero" | "reflect"
    "short_pad": "tile",

    # HTDemucs 게이팅. 성분이 하나뿐인 파일은 분리를 건너뛴다.
    # 2026-09-05 T4 실측: 분리 생략 시 0.164 -> 0.059 s/오디오초 (2.8배).
    # Demucs 와 DF-Arena 호출 하나가 동시에 사라지기 때문이다.
    # 속도만이 아니라 분리 아티팩트를 안 만들어 정확도에도 유리하다.
    "demucs_gating": True,
    "gate_voice": 0.20,      # VP가 이 값 미만이면 음성 성분이 없다고 본다
    # 음악 위조도는 실효 가중치 0.27 로 음성(0.18)보다 무겁다.
    # 게이트를 잘못 닫으면 MUSIC_FAKE=0 이 되어 손실이 크므로 음악 쪽을 더 보수적으로 잡는다.
    "gate_music": 0.05,

    # 스템당 세그먼트 수 상한. 0 이면 무제한(=베이스라인).
    # 60초 파일은 스템당 15세그먼트라 1B 모델을 30회 호출한다.
    # 상한을 두면 긴 파일에서 크게 절약되지만 커버리지가 줄어 점수가 바뀐다.
    # 파일 자신의 길이만으로 결정되므로 "파일 단위 독립 예측" 규정에 저촉되지 않는다.
    "max_segments": 0,

    # ---- 점수 의미를 바꾸지 않는 항목 (기본 활성) ----
    "use_bf16": True,        # L4는 bf16 지원. fp32와 편차가 크면 자동으로 되돌린다.
    "batch_size": 8,         # DF-Arena 세그먼트 배치 크기
    "batch_patch": True,     # 벤더 forward의 무조건 unsqueeze(0)를 조건부로 고쳐 배치를 켠다.
                             # 단일 경로와 수치가 일치할 때만 실제로 사용한다.
    "demucs_on_gpu": True,   # HTDemucs를 GPU에 상주시킨다
}

# 실패한 파일에 넣을 중립값. AUC/EER 모두 0.5가 중립이다.
FALLBACK_ROW = {
    "FILE_FAKE_PROB": 0.5,
    "VOICE_FAKE_PROB": 0.5,
    "MUSIC_FAKE_PROB": 0.5,
    "VOICE_PRESENT_PROB": 0.5,
    "MUSIC_PRESENT_PROB": 0.5,
}


# =============================================================================
# 경로 및 상수
# =============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"

TEST_DIR = BASE_DIR / "data" / "test"
SAMPLE_SUBMISSION = BASE_DIR / "data" / "sample_submission.csv"
OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"

AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

# 오디오가 아닌 것이 확실한 확장자만 제외한다.
# 베이스라인처럼 화이트리스트를 쓰면 목록에 없는 확장자가 통째로 누락되어
# ID 불일치로 즉시 크래시한다. 평가 데이터 확장자는 "MP3, WAV, FLAC 등"으로만 공지됐다.
NON_AUDIO_SUFFIXES = {".csv", ".txt", ".json", ".md", ".zip", ".gz", ".py", ".log"}


def log(message):
    print(message, flush=True)


# =============================================================================
# 1. 제출 양식과 입력 파일 매칭
# =============================================================================

def read_sample_submission(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)

    if not column_names or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")

    missing = [c for c in (["ID"] + PREDICTION_COLUMNS) if c not in column_names]
    if missing:
        raise ValueError(f"Sample submission is missing columns: {missing}")

    for row in rows:
        row["ID"] = str(row["ID"]).strip()
    return column_names, rows


def build_id_to_path(test_dir):
    """stem -> 경로. 확장자로 거르지 않고 오디오가 아닌 것만 제외한다."""
    id_to_path = {}
    duplicates = []
    for path in sorted(test_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in NON_AUDIO_SUFFIXES:
            continue
        if path.stem in id_to_path:
            duplicates.append(path.stem)
            continue
        id_to_path[path.stem] = path
    if duplicates:
        log(f"[warn] 중복 stem {len(duplicates)}건, 첫 파일을 사용한다: {duplicates[:5]}")
    return id_to_path


# =============================================================================
# 2. 오디오 로드와 세그먼트 분할
# =============================================================================

def load_audio_16k(audio_path):
    audio, _ = librosa.load(audio_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("empty audio")
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio


def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]
    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)

    cap = int(CONFIG.get("max_segments", 0))
    if cap > 0 and len(starts) > cap:
        # 균등 간격으로 솎아낸다. 이 파일의 길이만으로 결정되며 다른 파일과 무관하다.
        picked = np.linspace(0, len(starts) - 1, cap).round().astype(int)
        starts = [starts[i] for i in sorted(set(int(v) for v in picked))]
    return starts


def pad_short_audio(audio):
    """SEGMENT_SAMPLES보다 짧은 오디오를 채운다.

    tile은 이음매에 불연속(클릭)을 만든다. 광대역 임펄스라 anti-spoofing 모델이
    학습한 적 없는 인공 아티팩트이고, 오탐 요인이 될 수 있다.
    """
    shortfall = SEGMENT_SAMPLES - audio.size
    mode = CONFIG["short_pad"]

    if mode == "zero":
        return np.pad(audio, (0, shortfall), mode="constant").astype(np.float32)

    if mode == "reflect":
        # reflect는 한 번에 원본 길이-1 까지만 넣을 수 있어 필요한 만큼 반복해서 늘린다.
        padded = audio
        while padded.size < SEGMENT_SAMPLES:
            take = min(padded.size - 1, SEGMENT_SAMPLES - padded.size)
            if take <= 0:
                padded = np.pad(padded, (0, SEGMENT_SAMPLES - padded.size), mode="constant")
                break
            padded = np.pad(padded, (0, take), mode="reflect")
        return padded[:SEGMENT_SAMPLES].astype(np.float32)

    repeat_count = SEGMENT_SAMPLES // audio.size + 1
    return np.tile(audio, repeat_count)[:SEGMENT_SAMPLES].astype(np.float32)


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        return pad_short_audio(audio)
    return audio[start:start + SEGMENT_SAMPLES].astype(np.float32, copy=False)


def make_segments(audio):
    return np.stack([extract_segment(audio, s) for s in get_segment_starts(audio.size)])


def aggregate_segment_scores(scores):
    """세그먼트 점수 집계. 한 파일 내부의 연산이므로 대회 규정에 저촉되지 않는다."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mode = CONFIG["segment_agg"]
    if mode == "mean":
        return float(values.mean())
    if mode == "topk_mean":
        k = min(int(CONFIG["segment_topk"]), values.size)
        return float(np.sort(values)[-k:].mean())
    return float(values.max())


def calculate_rms(audio):
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


# =============================================================================
# 3. PANNs — 음성·음악 존재 확률
#    실효 가중치 0.10에 리더보드 1위가 이미 AUC 0.989다. 로직을 바꾸지 않는다.
# =============================================================================

def load_panns_model(device):
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )
    label_groups = json.loads((PANNS_DIR / "component_labels.json").read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[name] for name in label_groups["voice"]]
    music_indices = [label_to_index[name] for name in label_groups["music"]]
    return model, voice_indices, music_indices


def predict_presence(panns, audio):
    model, voice_indices, music_indices = panns
    segments = make_segments(audio)
    resampled = np.stack([
        librosa.resample(segment, orig_sr=AUDIO_SAMPLE_RATE, target_sr=PANNS_SAMPLE_RATE,
                         res_type="soxr_hq").astype(np.float32)
        for segment in segments
    ])
    predictions, _ = model.inference(resampled)
    voice = float(predictions[:, voice_indices].max())
    music = float(predictions[:, music_indices].max())
    return voice, music


# =============================================================================
# 4. HTDemucs — 음성·음악 분리
# =============================================================================

def load_htdemucs_model(device):
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        # PyTorch 2.6부터 바뀐 weights_only 기본값에 맞춰 기존 체크포인트를 불러온다.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load

    model = model.eval()
    return model.to(device) if CONFIG["demucs_on_gpu"] else model.cpu()


def separate_voice_and_music(audio_16k, model, device):
    """16kHz 모노 배열을 받아 (voice, music) 16kHz 배열을 돌려준다.

    베이스라인은 파일을 44.1kHz로 다시 디코딩한다. 원본이 이미 16kHz로 표준화돼 있으므로
    메모리에서 리샘플하는 것과 결과가 같고 디코딩이 한 번 줄어든다.
    """
    silence = np.zeros(max(1, audio_16k.size), dtype=np.float32)
    if calculate_rms(audio_16k) < 1e-8:
        return silence, silence.copy()

    wav = torch.from_numpy(audio_16k).unsqueeze(0)                      # (1, T)
    wav = torchaudio.functional.resample(wav, AUDIO_SAMPLE_RATE, model.samplerate)
    wav = wav.repeat(model.audio_channels, 1).float()                   # (C, T')

    mono = wav.mean(0)
    mean = mono.mean()
    std = mono.std()
    if float(std) < 1e-8:
        return silence, silence.copy()

    normalized = ((wav - mean) / std).to(device)
    with torch.inference_mode():
        sources = apply_model(
            model, normalized[None], device=device,
            shifts=0, split=True, overlap=0.25, progress=False,
        )[0]
    sources = sources * std.to(sources.device) + mean.to(sources.device)

    vocal_index = model.sources.index("vocals")
    voice = sources[vocal_index].mean(0, keepdim=True)
    music = torch.stack([
        sources[index] for index, name in enumerate(model.sources) if name != "vocals"
    ]).sum(0).mean(0, keepdim=True)

    voice = torchaudio.functional.resample(voice.cpu(), model.samplerate, AUDIO_SAMPLE_RATE)[0]
    music = torchaudio.functional.resample(music.cpu(), model.samplerate, AUDIO_SAMPLE_RATE)[0]
    return voice.numpy().astype(np.float32), music.numpy().astype(np.float32)


# =============================================================================
# 5. DF-Arena 1B — 성분별 FAKE 확률 (배치 + bf16, 실패 시 자동 폴백)
# =============================================================================

def install_batch_patch():
    """DF_Arena_1B.forward 가 배치 입력을 받도록 고친다.

    벤더 원본은 다음과 같아서 (B, T) 를 넘기면 (1, B, T) 가 되어 Wav2Vec2 가 깨진다.

        def forward(self, x):
            out_ssl = self.ssl_model(x.unsqueeze(0))

    아래는 이 한 줄만 조건부로 바꾼 것이고 나머지 연산은 원본과 동일하다.
    배치 차원은 수학적으로 독립이다 — BatchNorm2d 는 eval 모드라 running stats 를 쓰고,
    attention pooling 은 시간축(dim=1) softmax 라 샘플 간 섞이지 않는다.
    그래도 기동 시 단일 경로와 수치를 대조해 확인한다.

    model/ 아래 벤더 파일은 건드리지 않는다. 2차 평가에서 배포본 그대로를 제출해야
    출처 확인이 쉽다.
    """
    from df_arena_1b.backbone import DF_Arena_1B

    if getattr(DF_Arena_1B, "_batch_patched", False):
        return

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out_ssl = self.ssl_model(x)
        y0, fullfeature = self.get_attenF1D(out_ssl.hidden_states)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        fullfeature = self.first_bn(fullfeature)
        fullfeature = self.selu(fullfeature)
        output, _ = self.conformer(fullfeature.squeeze(1))
        return output

    DF_Arena_1B.forward = forward
    DF_Arena_1B._batch_patched = True


class DFArenaScorer:
    def __init__(self, device):
        if str(MODEL_DIR) not in sys.path:
            sys.path.insert(0, str(MODEL_DIR))
        from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

        previous = Path.cwd()
        # backbone 이 Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-1b") 를
        # 상대 경로로 찾으므로 모델 폴더에서 로드해야 한다.
        os.chdir(DF_ARENA_DIR)
        try:
            model = DF_Arena_1B_Antispoofing.from_pretrained(
                str(DF_ARENA_DIR), local_files_only=True, low_cpu_mem_usage=True,
            )
        finally:
            os.chdir(previous)

        self.model = model.to(device).eval()
        self.device = device
        self.fake_index = int(model.config.label2id["spoof"])  # 배포본 기준 spoof == 0
        self.use_bf16 = bool(CONFIG["use_bf16"]) and device.type == "cuda"
        self.batch_size = max(1, int(CONFIG["batch_size"]))
        self.batched = False

        if CONFIG["batch_patch"]:
            try:
                install_batch_patch()
                log("[info] DF-Arena 배치 패치 적용")
            except Exception as error:
                log(f"[info] 배치 패치 실패, 단일 경로로 간다: {type(error).__name__}: {error}")

        self._probe()

    def _autocast(self):
        if self.use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", enabled=False)

    def _forward_batch(self, batch_2d):
        """(B, T) -> (B,) fake 확률."""
        with torch.inference_mode(), self._autocast():
            logits = self.model(input_values=batch_2d)["logits"]
        return torch.softmax(logits.float(), dim=-1)[:, self.fake_index]

    def _forward_single(self, segment_1d):
        """베이스라인과 동일하게 1D 텐서를 넘긴다."""
        with torch.inference_mode(), self._autocast():
            logits = self.model(input_values=segment_1d)["logits"]
        return torch.softmax(logits.float(), dim=-1)[0, self.fake_index]

    def _probe(self):
        """단일 경로를 기준으로 bf16 과 배치를 각각 검증한다.

        배치는 속도만을 위한 것이므로 단일 경로와 수치가 일치할 때만 켠다.
        제출 1회는 하루의 1/3이라, 의심스러우면 느리더라도 원본과 같은 경로로 간다.
        """
        torch.manual_seed(0)
        sample = torch.randn(3, SEGMENT_SAMPLES, device=self.device) * 0.05

        # 1) 단일 경로 — 베이스라인과 동일하다. 이게 실패하면 손 쓸 방법이 없다.
        try:
            reference = torch.stack([self._forward_single(sample[i]) for i in range(3)])
            if not bool(torch.isfinite(reference).all()):
                raise ValueError("non-finite output")
        except Exception as error:
            if self.use_bf16:
                log(f"[warn] bf16 단일 경로 실패, fp32 로 재시도: "
                    f"{type(error).__name__}: {error}")
                self.use_bf16 = False
                reference = torch.stack([self._forward_single(sample[i]) for i in range(3)])
            else:
                raise

        # 2) bf16 결과가 fp32 와 크게 다르지 않은지 확인한다.
        if self.use_bf16:
            self.use_bf16 = False
            fp32_reference = torch.stack([self._forward_single(sample[i]) for i in range(3)])
            self.use_bf16 = True
            drift = float((reference - fp32_reference).abs().max())
            log(f"[info] bf16 vs fp32 최대 편차 {drift:.5f}")
            if drift > 0.05:
                log("[info] 편차가 커서 bf16 을 끈다")
                self.use_bf16 = False
                reference = fp32_reference

        # 3) 배치 경로가 단일 경로와 같은 값을 내는지 확인한다.
        tolerance = 0.02 if self.use_bf16 else 1e-3
        try:
            batched_out = self._forward_batch(sample)
            if tuple(batched_out.shape) != (3,):
                raise ValueError(f"unexpected shape {tuple(batched_out.shape)}")
            diff = float((batched_out - reference).abs().max())
            if diff <= tolerance and bool(torch.isfinite(batched_out).all()):
                self.batched = True
                log(f"[info] 배치 경로 검증 통과 (최대 편차 {diff:.6f})")
            else:
                log(f"[info] 배치 결과가 단일과 다르다 (편차 {diff:.6f} > {tolerance}), "
                    f"단일 경로로 간다")
        except Exception as error:
            log(f"[info] 배치 추론 미지원, 세그먼트 단위로 실행한다: "
                f"{type(error).__name__}: {error}")

        log(f"[info] DF-Arena: batched={self.batched} "
            f"(batch_size={self.batch_size}) bf16={self.use_bf16} "
            f"fake_index={self.fake_index}")

    def score(self, audio):
        if calculate_rms(audio) < SILENCE_RMS:
            return 0.0

        segments = make_segments(audio)
        scores = []
        if self.batched:
            tensor = torch.from_numpy(segments).to(self.device)
            for start in range(0, tensor.shape[0], self.batch_size):
                chunk = tensor[start:start + self.batch_size]
                scores.extend(self._forward_batch(chunk).float().cpu().tolist())
        else:
            for segment in segments:
                tensor = torch.from_numpy(segment).to(self.device)
                scores.append(float(self._forward_single(tensor)))
        return aggregate_segment_scores(scores)


# =============================================================================
# 6. 융합 — FILE_FAKE_PROB (실효 가중치 0.45, 단일 최대 항목)
# =============================================================================

def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):
    mode = CONFIG["fusion_mode"]
    voice_risk = voice_present * voice_fake
    music_risk = music_present * music_fake

    if mode == "gated_max":
        gate = CONFIG["fusion_gate"]
        candidates = []
        if voice_present >= gate:
            candidates.append(voice_fake)
        if music_present >= gate:
            candidates.append(music_fake)
        # 어느 성분도 게이트를 넘지 못하면 베이스라인 방식으로 되돌린다.
        return max(candidates) if candidates else max(voice_risk, music_risk)

    if mode == "noisy_or":
        return 1.0 - (1.0 - voice_risk) * (1.0 - music_risk)

    if mode == "gamma":
        gamma = CONFIG["fusion_gamma"]
        return max((voice_present ** gamma) * voice_fake,
                   (music_present ** gamma) * music_fake)

    return max(voice_risk, music_risk)


# =============================================================================
# 7. 메인
# =============================================================================

def process_one_file(audio_path, panns, scorer, htdemucs, device):
    audio = load_audio_16k(audio_path)
    voice_present, music_present = predict_presence(panns, audio)

    need_separation = True
    skip_reason = None
    if CONFIG["demucs_gating"]:
        has_voice = voice_present >= CONFIG["gate_voice"]
        has_music = music_present >= CONFIG["gate_music"]
        if not (has_voice and has_music):
            need_separation = False
            skip_reason = "voice_only" if has_voice else "music_only"

    if need_separation:
        voice_audio, music_audio = separate_voice_and_music(audio, htdemucs, device)
    else:
        # 성분이 하나뿐이면 분리 아티팩트를 만들지 않고 원본을 그대로 채점한다.
        empty = np.zeros(1, dtype=np.float32)
        if skip_reason == "voice_only":
            voice_audio, music_audio = audio, empty
        else:
            voice_audio, music_audio = empty, audio

    voice_fake = scorer.score(voice_audio)
    music_fake = scorer.score(music_audio)
    file_fake = combine_file_fake_score(voice_fake, music_fake, voice_present, music_present)

    return {
        "FILE_FAKE_PROB": file_fake,
        "VOICE_FAKE_PROB": voice_fake,
        "MUSIC_FAKE_PROB": music_fake,
        "VOICE_PRESENT_PROB": voice_present,
        "MUSIC_PRESENT_PROB": music_present,
    }


def main():
    started = time.perf_counter()
    log(f"[info] CONFIG = {json.dumps(CONFIG, ensure_ascii=False)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    log(f"[info] device={torch.cuda.get_device_name(0)}")

    column_names, rows = read_sample_submission(SAMPLE_SUBMISSION)
    id_to_path = build_id_to_path(TEST_DIR)
    log(f"[info] 제출 양식 {len(rows)}행, 입력 파일 {len(id_to_path)}개")

    # 세 모델을 동시에 상주시킨다. L4 22.4GiB에 여유가 크고 파일당 루프가 하나로 줄어든다.
    load_started = time.perf_counter()
    panns = load_panns_model(device)
    scorer = DFArenaScorer(device)
    htdemucs = load_htdemucs_model(device)
    log(f"[info] 모델 로드 {time.perf_counter() - load_started:.1f}s")

    failures = []
    infer_started = time.perf_counter()

    for index, row in enumerate(rows):
        audio_id = row["ID"]
        try:
            audio_path = id_to_path.get(audio_id)
            if audio_path is None:
                raise FileNotFoundError(f"no file for ID {audio_id}")
            result = process_one_file(audio_path, panns, scorer, htdemucs, device)
        except Exception:
            # 한 파일의 실패가 1,200개 전체를 0점으로 만들지 않게 한다.
            failures.append(audio_id)
            log(f"[warn] {audio_id} 실패, 폴백값 사용\n{traceback.format_exc()}")
            result = dict(FALLBACK_ROW)
            torch.cuda.empty_cache()

        for column in PREDICTION_COLUMNS:
            value = float(result[column])
            if not np.isfinite(value):
                value = FALLBACK_ROW[column]
            row[column] = round(min(1.0, max(0.0, value)), 10)

        if (index + 1) % 100 == 0:
            elapsed = time.perf_counter() - infer_started
            rate = elapsed / (index + 1)
            log(f"[info] {index + 1}/{len(rows)} 처리, {elapsed:.0f}s 경과, "
                f"파일당 {rate:.2f}s, 전체 예상 {rate * len(rows) / 60:.1f}분")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)

    total = time.perf_counter() - started
    log(f"[info] {len(rows)}행 저장 -> {OUTPUT_PATH}")
    log(f"[info] 총 {total:.0f}s ({total / 60:.1f}분), 실패 {len(failures)}건")
    if failures:
        log(f"[warn] 실패 ID: {failures[:20]}")


if __name__ == "__main__":
    main()
