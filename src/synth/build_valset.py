#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""로컬 검증셋을 합성한다.

대회는 학습·검증 데이터를 제공하지 않는다. 리더보드는 하루 3회뿐이므로
설정 비교를 할 측정 대상이 필요하다. 평가 데이터의 성질을 재현해 만든다.

재현하는 성질 (대회 명세):
  - 유형: 음성 단독 / 음악 단독 / 혼합 (동시 또는 순차)
  - 길이 4~60초, 16 kHz
  - 포맷 MP3 / WAV / FLAC 혼재, 모노·스테레오 혼재
  - 일부 전화채널
  - 보컬은 음성 성분이다
  - 품질개선·잡음제거·음량조정 등 성분을 새로 생성하지 않는 후처리만 적용된 경우 REAL

마지막 줄이 중요하다. REAL 원본에 일부러 후처리를 걸어 REAL 로 남기는 샘플을 넣는다.
그래야 모델이 "후처리 아티팩트 = FAKE" 라는 지름길을 학습했는지 드러난다.
"""

import csv
import random
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
LABEL_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

# 구성 비율. 혼합에서 한쪽 성분만 FAKE 인 경우가 이 대회의 어려운 지점이라 비중을 둔다.
COMPOSITIONS = [
    # (이름, 음성있음, 음악있음, 음성가짜, 음악가짜, 가중치)
    ("voice_real",        True,  False, False, False, 12),
    ("voice_fake",        True,  False, True,  False, 12),
    ("music_real",        False, True,  False, False, 10),
    ("music_fake",        False, True,  False, True,  10),
    ("mix_real_real",     True,  True,  False, False, 12),
    ("mix_fakeV_realM",   True,  True,  True,  False, 12),   # 음성만 가짜
    ("mix_realV_fakeM",   True,  True,  False, True,  12),   # 음악만 가짜
    ("mix_fake_fake",     True,  True,  True,  True,  10),
]


def rms(audio):
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def load_16k_mono(path, rng, min_seconds=1.0):
    """어떤 포맷이든 16 kHz 모노로 읽는다. 너무 짧으면 None."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        # 의존성을 늘리지 않으려고 선형 보간으로 리샘플한다. 검증셋 생성용이라 충분하다.
        target_len = int(round(audio.size * SAMPLE_RATE / sr))
        if target_len < 2:
            return None
        audio = np.interp(np.linspace(0, audio.size - 1, target_len),
                          np.arange(audio.size), audio).astype(np.float32)
    if audio.size < int(min_seconds * SAMPLE_RATE) or rms(audio) < 1e-5:
        return None
    return audio


def take_span(audio, seconds, rng):
    """필요한 길이를 뽑는다. 짧으면 이어붙이되 이음매에 짧은 크로스페이드를 준다."""
    need = int(seconds * SAMPLE_RATE)
    if audio.size >= need:
        start = rng.integers(0, audio.size - need + 1) if audio.size > need else 0
        return audio[start:start + need].copy()

    fade = min(int(0.01 * SAMPLE_RATE), audio.size // 4)
    out = np.zeros(need, dtype=np.float32)
    pos = 0
    while pos < need:
        chunk = audio[: min(audio.size, need - pos)]
        if pos > 0 and fade > 0 and chunk.size > fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            out[pos:pos + fade] *= (1.0 - ramp)
            out[pos:pos + fade] += chunk[:fade] * ramp
            out[pos + fade:pos + chunk.size] = chunk[fade:]
        else:
            out[pos:pos + chunk.size] = chunk
        pos += chunk.size
    return out


def normalize(audio, target_rms=0.05):
    current = rms(audio)
    if current < 1e-9:
        return audio
    return (audio * (target_rms / current)).astype(np.float32)


def mix_sequential(voice, music, rng):
    """순차 혼합 — 앞은 음악, 뒤는 음성 같은 구성. 대회 정의에 명시돼 있다."""
    total = max(voice.size, music.size)
    out = np.zeros(total, dtype=np.float32)
    cut = int(total * rng.uniform(0.3, 0.7))
    out[:cut] = take_span(music, cut / SAMPLE_RATE, rng)
    out[cut:] = take_span(voice, (total - cut) / SAMPLE_RATE, rng)
    return out


def mix_overlap(voice, music, rng, snr_db=None):
    """동시 혼합. 음성 대 음악 SNR 을 무작위로 준다."""
    length = max(voice.size, music.size)
    voice = take_span(voice, length / SAMPLE_RATE, rng)
    music = take_span(music, length / SAMPLE_RATE, rng)
    snr_db = rng.uniform(-5, 15) if snr_db is None else snr_db
    scale = rms(voice) / max(rms(music), 1e-9) / (10 ** (snr_db / 20))
    return (voice + music * scale).astype(np.float32)


def apply_benign_postprocess(audio, rng):
    """성분을 새로 생성하지 않는 후처리. 대회 규정상 라벨은 REAL 을 유지한다."""
    out = audio.copy()
    out *= float(rng.uniform(0.5, 1.8))                       # 음량 조정
    if rng.random() < 0.5:                                     # 완만한 하이패스 (잡음 제거 흉내)
        kernel = np.array([1.0, -0.95], dtype=np.float32)
        out = np.convolve(out, kernel, mode="same").astype(np.float32)
    peak = float(np.abs(out).max())
    if peak > 0.99:
        out = (out / peak * 0.99).astype(np.float32)
    return out


def _fir_lowpass(cutoff_hz, num_taps=127):
    """윈도우드 싱크 저역통과. scipy 의존성을 늘리지 않으려고 직접 만든다."""
    fc = cutoff_hz / SAMPLE_RATE
    n = np.arange(num_taps) - (num_taps - 1) / 2.0
    taps = 2 * fc * np.sinc(2 * fc * n) * np.hamming(num_taps)
    return (taps / taps.sum()).astype(np.float32)


def to_telephone(audio, rng, low_hz=300.0, high_hz=3400.0):
    """전화채널 대역제한 (300~3400 Hz).

    주의: 안티에일리어싱 없이 데시메이션하면 고역이 제거되는 게 아니라 되접혀 들어온다.
    실제로 첫 구현이 그랬고 자체 검사에서 걸렸다. 반드시 먼저 저역통과를 건다.
    """
    lowpass = _fir_lowpass(high_hz)
    band = np.convolve(audio, lowpass, mode="same").astype(np.float32)

    # 300 Hz 아래를 깎는다. 저역통과 두 개의 차로 고역통과를 만든다.
    band = (band - np.convolve(band, _fir_lowpass(low_hz), mode="same")).astype(np.float32)

    # 대역제한 뒤에는 8 kHz 왕복이 안전하다 (에일리어싱될 성분이 이미 없다)
    down = np.interp(np.linspace(0, band.size - 1, max(2, band.size // 2)),
                     np.arange(band.size), band).astype(np.float32)
    up = np.interp(np.linspace(0, down.size - 1, band.size),
                   np.arange(down.size), down).astype(np.float32)

    # 업샘플의 선형보간은 8 kHz 부근에 스펙트럼 이미지를 남긴다.
    # 실제 전화망도 재구성 필터를 거치므로 한 번 더 저역통과한다.
    return np.convolve(up, lowpass, mode="same").astype(np.float32)


def write_with_format(audio, out_path, fmt, stereo, rng):
    """포맷·채널을 적용해 저장한다. mp3 는 ffmpeg 왕복으로 실제 손실압축을 건다."""
    data = audio
    if stereo:
        # 좌우에 미세한 이득 차이를 준다. 평가셋에 스테레오가 섞여 있다.
        data = np.stack([audio, audio * float(rng.uniform(0.75, 1.0))], axis=1)

    if fmt == "mp3":
        tmp = out_path.with_suffix(".tmp.wav")
        sf.write(tmp, data, SAMPLE_RATE)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(tmp),
             "-b:a", rng.choice(["64k", "96k", "128k"]), str(out_path)],
            check=True,
        )
        tmp.unlink(missing_ok=True)
    else:
        sf.write(out_path, data, SAMPLE_RATE, format=fmt.upper())


def build(sources, out_dir, count=400, seed=0, formats=("wav", "mp3", "flac"),
          telephone_ratio=0.20, stereo_ratio=0.25, postprocess_ratio=0.30):
    """검증셋을 만든다.

    sources: {"voice_real": [경로...], "voice_fake": [...],
              "music_real": [...], "music_fake": [...]}
    가짜 음악 소스가 없으면 음악 FAKE 구성을 자동으로 건너뛴다 (Music EER 은 nan 이 된다).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    out_dir = Path(out_dir)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    available = {key: list(paths) for key, paths in sources.items() if paths}
    for key in ("voice_real", "voice_fake", "music_real"):
        assert available.get(key), f"소스가 비었다: {key}"
    has_fake_music = bool(available.get("music_fake"))
    if not has_fake_music:
        print("[warn] music_fake 소스가 없다. 음악 FAKE 구성을 제외한다 -> Music EER 계산 불가")

    compositions = [c for c in COMPOSITIONS if has_fake_music or not c[4]]
    weights = np.array([c[5] for c in compositions], dtype=np.float64)
    weights /= weights.sum()

    rows = []
    made = 0
    attempts = 0
    while made < count and attempts < count * 20:
        attempts += 1
        name, has_voice, has_music, voice_fake, music_fake = \
            compositions[rng.choice(len(compositions), p=weights)][:5]

        seconds = float(rng.uniform(4.0, 60.0))

        voice = music = None
        if has_voice:
            key = "voice_fake" if voice_fake else "voice_real"
            voice = load_16k_mono(random.choice(available[key]), rng)
            if voice is None:
                continue
            voice = normalize(take_span(voice, seconds, rng))
        if has_music:
            key = "music_fake" if music_fake else "music_real"
            music = load_16k_mono(random.choice(available[key]), rng)
            if music is None:
                continue
            music = normalize(take_span(music, seconds, rng), target_rms=0.04)

        if has_voice and has_music:
            audio = (mix_sequential(voice, music, rng) if rng.random() < 0.25
                     else mix_overlap(voice, music, rng))
        else:
            audio = voice if has_voice else music

        # REAL 성분만 있는 파일에 한해 후처리를 걸고 라벨은 REAL 로 둔다
        if not (voice_fake or music_fake) and rng.random() < postprocess_ratio:
            audio = apply_benign_postprocess(audio, rng)
        if rng.random() < telephone_ratio:
            audio = to_telephone(audio, rng)

        peak = float(np.abs(audio).max())
        if peak > 0.99:
            audio = (audio / peak * 0.99).astype(np.float32)
        if rms(audio) < 1e-5:
            continue

        fmt = str(rng.choice(list(formats)))
        stereo = bool(rng.random() < stereo_ratio)
        audio_id = f"VAL_{made:05d}"
        path = out_dir / "test" / f"{audio_id}.{fmt}"
        try:
            write_with_format(audio, path, fmt, stereo, rng)
        except Exception as error:
            print(f"[warn] 저장 실패 {audio_id} ({fmt}): {error}")
            continue

        rows.append({
            "ID": audio_id,
            "composition": name,
            "seconds": round(audio.size / SAMPLE_RATE, 2),
            "format": fmt,
            "stereo": int(stereo),
            "FILE_FAKE_PROB": int(voice_fake or music_fake),
            "VOICE_FAKE_PROB": int(voice_fake),
            "MUSIC_FAKE_PROB": int(music_fake),
            "VOICE_PRESENT_PROB": int(has_voice),
            "MUSIC_PRESENT_PROB": int(has_music),
        })
        made += 1

    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 파이프라인이 그대로 읽을 수 있게 제출 양식도 같이 만든다
    with (out_dir / "sample_submission.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ID"] + LABEL_COLUMNS)
        for row in rows:
            writer.writerow([row["ID"]] + [0.5] * 5)

    print(f"검증셋 {len(rows)}개 -> {out_dir}")
    counts = {}
    for row in rows:
        counts[row["composition"]] = counts.get(row["composition"], 0) + 1
    for name in sorted(counts):
        print(f"  {name:18s} {counts[name]:4d}")
    return rows
