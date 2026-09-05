#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/eval 의 측정 도구와 스윕 엔진을 GPU 없이 검증한다.

precompute 만 GPU 가 필요하고 나머지는 전부 CPU 에서 돌릴 수 있다.
캐시를 손으로 만들어 넣으면 스윕 경로 전체를 여기서 확인할 수 있다.
"""

import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from selftest_logic import install_stubs          # noqa: E402  스텁 재사용
from eval import sweep as sweep_mod               # noqa: E402
from eval.metric import evaluate, score_delta_explained  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def make_cache_and_labels(n=240, seed=0):
    """정답을 알고 있는 인공 캐시를 만든다.

    '좋은 음성 점수'와 '쓸모없는 음악 점수'를 일부러 넣어,
    스윕이 그 차이를 실제로 잡아내는지 본다 — 지금 리더보드에서 의심하는 상황이다.
    """
    rng = np.random.default_rng(seed)
    cache, labels = [], []

    for i in range(n):
        kind = i % 4
        has_voice = kind in (0, 2, 3)
        has_music = kind in (1, 2, 3)
        voice_fake = has_voice and (i // 4) % 2 == 0
        music_fake = has_music and (i // 8) % 2 == 0

        vp = float(rng.uniform(0.75, 0.99)) if has_voice else float(rng.uniform(0.0, 0.08))
        mp = float(rng.uniform(0.75, 0.99)) if has_music else float(rng.uniform(0.0, 0.08))

        # 음성 헤드는 잘 맞는다 (분리가 큼)
        def voice_scores():
            center = 0.85 if voice_fake else 0.15
            return list(np.clip(rng.normal(center, 0.08, 3), 0, 1))

        # 음악 헤드는 무작위다 — 도메인 밖 사용을 흉내낸다
        def music_scores():
            return list(rng.uniform(0, 1, 3))

        # 원본 전체 점수는 "하나라도 FAKE 면 FAKE" 를 꽤 잘 반영한다
        file_fake = int(voice_fake or music_fake)
        original = list(np.clip(rng.normal(0.80 if file_fake else 0.20, 0.10, 3), 0, 1))

        cache.append({
            "ID": f"VAL_{i:05d}", "vp": vp, "mp": mp,
            "original_segments": original,
            "voice_segments": voice_scores() if has_voice else list(rng.uniform(0, 0.2, 3)),
            "music_segments": music_scores() if has_music else list(rng.uniform(0, 0.2, 3)),
        })
        labels.append({
            "ID": f"VAL_{i:05d}",
            "FILE_FAKE_PROB": file_fake,
            "VOICE_FAKE_PROB": int(voice_fake),
            "MUSIC_FAKE_PROB": int(music_fake),
            "VOICE_PRESENT_PROB": int(has_voice),
            "MUSIC_PRESENT_PROB": int(has_music),
        })
    return cache, labels


def test_valset_signal_functions():
    """검증셋 합성의 순수 신호처리 부분을 검증한다. soundfile 은 스텁으로 대체한다."""
    import types
    if "soundfile" not in sys.modules:
        stub = types.ModuleType("soundfile")
        stub.read = lambda *a, **k: (np.zeros((16000, 1), dtype=np.float32), 16000)
        stub.write = lambda *a, **k: None
        sys.modules["soundfile"] = stub

    from synth import build_valset as bv
    rng = np.random.default_rng(0)
    sr = bv.SAMPLE_RATE

    print("\n[검증셋 합성 — 신호처리]")
    tone = np.sin(np.linspace(0, 200, 3 * sr)).astype(np.float32)

    span = bv.take_span(tone, 10.0, rng)
    check("긴 길이 요청 시 정확한 샘플 수", span.size == 10 * sr, str(span.size))
    check("이어붙여도 유한", bool(np.isfinite(span).all()))
    short = bv.take_span(tone, 1.0, rng)
    check("짧은 길이 요청 시 잘라내기", short.size == sr, str(short.size))

    loud = bv.normalize(tone * 0.001, target_rms=0.05)
    check("정규화가 목표 RMS 를 맞춘다", abs(bv.rms(loud) - 0.05) < 1e-6, str(bv.rms(loud)))
    check("무음 정규화가 폭주하지 않음",
          bool(np.isfinite(bv.normalize(np.zeros(1000, dtype=np.float32))).all()))

    # 광대역 신호로 재야 대역제한이 실제로 걸렸는지 알 수 있다.
    # 저주파 정현파로 재면 고역 자체가 없어 아무것도 측정되지 않는다 (첫 시도가 그랬다).
    noise = rng.normal(0, 0.1, 3 * sr).astype(np.float32)
    tel = bv.to_telephone(noise, rng)
    check("전화채널이 길이를 보존", tel.size == noise.size, str(tel.size))

    def band_energy(signal, low_hz, high_hz):
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(signal.size, 1 / sr)
        return float(spectrum[(freqs >= low_hz) & (freqs < high_hz)].sum())

    pass_before = band_energy(noise, 500, 3000)
    pass_after = band_energy(tel, 500, 3000)
    stop_before = band_energy(noise, 4500, 8000)
    stop_after = band_energy(tel, 4500, 8000)

    check("통과대역(500~3000Hz)이 살아 있다", pass_after > 0.3 * pass_before,
          f"{pass_after:.0f} / {pass_before:.0f}")
    check("차단대역(4500Hz+)이 크게 줄었다", stop_after < 0.05 * stop_before,
          f"{stop_after:.1f} / {stop_before:.1f}")
    check("300Hz 아래도 줄었다",
          band_energy(tel, 0, 150) < 0.3 * band_energy(noise, 0, 150),
          f"{band_energy(tel, 0, 150):.1f} / {band_energy(noise, 0, 150):.1f}")

    post = bv.apply_benign_postprocess(tone, rng)
    check("후처리 출력이 유한하고 클리핑 없음",
          bool(np.isfinite(post).all()) and float(np.abs(post).max()) <= 0.995,
          str(float(np.abs(post).max())))

    voice = np.sin(np.linspace(0, 500, 5 * sr)).astype(np.float32)
    music = np.sin(np.linspace(0, 90, 4 * sr)).astype(np.float32)
    over = bv.mix_overlap(voice, music, rng)
    check("동시 혼합 길이 = 긴 쪽", over.size == max(voice.size, music.size), str(over.size))
    seq = bv.mix_sequential(voice, music, rng)
    check("순차 혼합 길이 = 긴 쪽", seq.size == max(voice.size, music.size), str(seq.size))
    check("혼합 결과가 유한", bool(np.isfinite(over).all()) and bool(np.isfinite(seq).all()))

    print("\n[구성 정의]")
    names = [c[0] for c in bv.COMPOSITIONS]
    check("혼합에서 한쪽만 FAKE 인 경우 포함",
          "mix_fakeV_realM" in names and "mix_realV_fakeM" in names, str(names))
    for name, has_v, has_m, v_fake, m_fake, _ in bv.COMPOSITIONS:
        ok = ((not v_fake) or has_v) and ((not m_fake) or has_m)
        check(f"라벨 정합성 {name}", ok)


def main():
    install_stubs()
    script = sweep_mod.load_script(REPO / "submit" / "script.py")
    cache, labels = make_cache_and_labels()

    print("[캐시 기반 예측]")
    ids, predictions = sweep_mod.predict(script, cache, {"file_head": "fusion"})
    check("모든 파일 예측", len(ids) == len(cache), f"{len(ids)}/{len(cache)}")
    check("5개 컬럼", set(predictions) == set(script.PREDICTION_COLUMNS))
    check("범위 [0,1]", all(bool(((v >= 0) & (v <= 1)).all()) for v in predictions.values()))
    check("CONFIG 원복", script.CONFIG["file_head"] == "direct",
          f"got {script.CONFIG['file_head']}")

    print("\n[정답 정렬]")
    truth = sweep_mod.build_truth(labels, ids)
    check("정답 길이 일치", len(truth["FILE_FAKE_PROB"]) == len(ids))
    check("정답이 0/1", set(np.unique(truth["FILE_FAKE_PROB"])) <= {0, 1})

    print("\n[지표 계산]")
    result = evaluate(predictions, truth)
    check("Score 가 [0,1]", 0 <= result["score"] <= 1, str(result["score"]))
    check("산식 일치",
          abs(result["score"] - (0.9 * result["ads"] + 0.1 * result["cps"])) < 1e-12)
    check("Voice/Music 표본 수가 전체보다 적다",
          result["n_voice"] < result["n"] and result["n_music"] < result["n"],
          f"{result['n_voice']}/{result['n_music']}/{result['n']}")

    print("\n[설계한 상황을 실제로 잡아내는가]")
    check("음악 EER 이 나쁘다 (무작위로 만들었다)", result["music_eer"] > 0.35,
          f"{result['music_eer']:.4f}")
    check("음성 EER 이 좋다 (분리되게 만들었다)", result["voice_eer"] < 0.15,
          f"{result['voice_eer']:.4f}")

    print("\n[스윕 — 오염된 음악 항을 우회하는 설정이 이겨야 한다]")
    results = sweep_mod.sweep(script, cache, labels,
                              [("baseline(fusion)", {"file_head": "fusion",
                                                     "fusion_mode": "baseline"}),
                               ("file:direct", {"file_head": "direct"}),
                               ("file:direct_max", {"file_head": "direct_max"})],
                              baseline_name="baseline(fusion)")
    names = [r["name"] for r in results]
    fusion = next(r for r in results if r["name"] == "baseline(fusion)")
    direct = next(r for r in results if r["name"] == "file:direct")
    check("direct 가 fusion 보다 file EER 이 낮다",
          direct["file_eer"] < fusion["file_eer"],
          f"direct {direct['file_eer']:.4f} vs fusion {fusion['file_eer']:.4f}")
    check("순위 1위가 direct 계열", names[0].startswith("file:direct"), names[0])

    parts = score_delta_explained(fusion, direct)
    check("분해 합이 총점 차이와 일치",
          abs(parts["total"] - (direct["score"] - fusion["score"])) < 1e-9,
          f"{parts['total']} vs {direct['score'] - fusion['score']}")
    check("차이가 file 항목에서 나왔다",
          abs(parts["file"]) > abs(parts["voice"]) + abs(parts["music"]),
          str({k: round(v, 5) for k, v in parts.items()}))

    print("\n[전체 기본 그리드가 오류 없이 돈다]")
    grid = sweep_mod.default_grid()
    ok = True
    for name, config in grid:
        try:
            ids2, preds2 = sweep_mod.predict(script, cache, config)
            assert len(ids2) == len(cache)
            assert all(bool(np.isfinite(v).all()) for v in preds2.values())
        except Exception as error:
            ok = False
            print(f"    {name}: {type(error).__name__}: {error}")
    check(f"기본 그리드 {len(grid)}개 전부 실행", ok)

    test_valset_signal_functions()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    print("precompute(GPU) 는 Kaggle 노트북에서만 검증된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
