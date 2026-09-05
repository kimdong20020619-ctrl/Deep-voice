#!/usr/bin/env python3
"""submit/script.py 의 순수 로직을 GPU 없이 검증한다.

로컬에 GPU도 torch도 없으므로 torch·librosa·torchaudio·demucs를 스텁으로 갈아끼우고
세그먼트 분할, 패딩, 점수 집계, 융합식, 파일 매칭만 실제로 돌린다.

리더보드 제출은 하루 3회뿐이고 런타임 오류는 그중 1회를 태운다.
GPU가 필요 없는 부분만이라도 여기서 걸러낸다.
"""

import csv
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "submit" / "script.py"


def install_stubs():
    """script.py가 import 하는 무거운 패키지를 최소 스텁으로 대체한다."""
    torch = types.ModuleType("torch")
    torch.load = lambda *a, **k: None
    torch.bfloat16 = "bfloat16"

    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def get_device_name(_index):
            return "stub"

    torch.cuda = _Cuda()
    torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False))
    torch.device = lambda name: types.SimpleNamespace(type=name)
    for name in ("from_numpy", "randn", "softmax", "stack", "isfinite", "inference_mode",
                 "autocast", "no_grad"):
        setattr(torch, name, lambda *a, **k: None)

    librosa = types.ModuleType("librosa")
    librosa.load = lambda *a, **k: (np.zeros(1, dtype=np.float32), 16000)
    librosa.resample = lambda y, **k: y

    torchaudio = types.ModuleType("torchaudio")
    torchaudio.functional = types.SimpleNamespace(resample=lambda w, *a, **k: w)

    demucs = types.ModuleType("demucs")
    demucs_apply = types.ModuleType("demucs.apply")
    demucs_apply.apply_model = lambda *a, **k: None
    demucs_pretrained = types.ModuleType("demucs.pretrained")
    demucs_pretrained.get_model = lambda *a, **k: None

    for name, module in [
        ("torch", torch), ("librosa", librosa), ("torchaudio", torchaudio),
        ("demucs", demucs), ("demucs.apply", demucs_apply),
        ("demucs.pretrained", demucs_pretrained),
    ]:
        sys.modules[name] = module


def load_script():
    install_stubs()
    spec = importlib.util.spec_from_file_location("submit_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def test_segment_starts(m):
    print("\n[세그먼트 분할]")
    n = m.SEGMENT_SAMPLES
    check("정확히 한 세그먼트 길이 -> [0]", m.get_segment_starts(n) == [0])
    check("4초(64000) < 64600 -> [0]", m.get_segment_starts(64_000) == [0])
    check("두 세그먼트 딱 맞음", m.get_segment_starts(2 * n) == [0, n])
    starts = m.get_segment_starts(100_000)
    check("끝이 남으면 끝에 맞춘 시작점 추가",
          starts == [0, 100_000 - n], f"got {starts}")
    starts = m.get_segment_starts(960_000)  # 1분
    check("1분 파일 세그먼트 수", len(starts) == 15, f"got {len(starts)}")
    check("마지막 세그먼트가 범위를 넘지 않음", starts[-1] + n == 960_000)


def test_segment_cap(m):
    print("\n[세그먼트 상한 max_segments]")
    n = m.SEGMENT_SAMPLES
    m.CONFIG["max_segments"] = 0
    full = m.get_segment_starts(960_000)          # 1분
    check("상한 0 -> 무제한(베이스라인)", len(full) == 15, f"got {len(full)}")

    m.CONFIG["max_segments"] = 8
    capped = m.get_segment_starts(960_000)
    check("상한 8 적용", len(capped) <= 8, f"got {len(capped)}")
    check("첫·끝 세그먼트 보존",
          capped[0] == full[0] and capped[-1] == full[-1], f"{capped[0]},{capped[-1]}")
    check("오름차순·중복 없음", capped == sorted(set(capped)))
    check("범위 이탈 없음", all(0 <= c <= 960_000 - n for c in capped))
    check("상한보다 짧으면 그대로", len(m.get_segment_starts(200_000)) == 4,
          str(len(m.get_segment_starts(200_000))))
    check("짧은 파일은 영향 없음", m.get_segment_starts(64_000) == [0])

    # 상한을 걸어도 세그먼트 추출이 깨지지 않아야 한다
    audio = np.zeros(960_000, dtype=np.float32)
    check("make_segments 모양 일치", m.make_segments(audio).shape == (len(capped), n))
    m.CONFIG["max_segments"] = 0


def test_gating_defaults(m):
    print("\n[게이팅 기본값 — 2026-09-05 실측 반영]")
    fresh = load_script()
    check("demucs_gating 기본 활성", fresh.CONFIG["demucs_gating"] is True)
    check("gate_music < gate_voice (음악 0.27 > 음성 0.18 가중치)",
          fresh.CONFIG["gate_music"] < fresh.CONFIG["gate_voice"],
          f"music={fresh.CONFIG['gate_music']} voice={fresh.CONFIG['gate_voice']}")
    check("max_segments 기본 0 (베이스라인 커버리지)", fresh.CONFIG["max_segments"] == 0)


def test_padding(m):
    print("\n[짧은 파일 패딩]")
    n = m.SEGMENT_SAMPLES
    audio = np.sin(np.linspace(0, 40, 64_000)).astype(np.float32)  # 정확히 4.0초

    for mode in ("tile", "zero", "reflect"):
        m.CONFIG["short_pad"] = mode
        padded = m.pad_short_audio(audio)
        check(f"{mode}: 길이 == SEGMENT_SAMPLES", padded.size == n, f"got {padded.size}")
        check(f"{mode}: dtype float32", padded.dtype == np.float32, str(padded.dtype))
        check(f"{mode}: 유한값", bool(np.isfinite(padded).all()))
        check(f"{mode}: 원본 앞부분 보존",
              bool(np.allclose(padded[:64_000], audio, atol=1e-6)))

    m.CONFIG["short_pad"] = "zero"
    check("zero: 꼬리가 실제로 0", float(np.abs(m.pad_short_audio(audio)[64_000:]).max()) == 0.0)

    # 극단값 — 샘플 1개짜리 오디오에서 reflect가 무한루프에 빠지지 않아야 한다
    m.CONFIG["short_pad"] = "reflect"
    tiny = m.pad_short_audio(np.array([0.5], dtype=np.float32))
    check("reflect: 1샘플 입력도 처리", tiny.size == n, f"got {tiny.size}")
    m.CONFIG["short_pad"] = "tile"

    print("\n[세그먼트 생성]")
    check("짧은 오디오 -> (1, N)", m.make_segments(audio).shape == (1, n))
    long_audio = np.zeros(200_000, dtype=np.float32)
    segments = m.make_segments(long_audio)
    check("긴 오디오 세그먼트 모양", segments.shape == (len(m.get_segment_starts(200_000)), n),
          str(segments.shape))
    check("경계를 넘는 슬라이스 없음", all(s.size == n for s in segments))


def test_aggregation(m):
    print("\n[세그먼트 집계]")
    scores = [0.1, 0.9, 0.4, 0.2]

    m.CONFIG["segment_agg"] = "max"
    check("max", abs(m.aggregate_segment_scores(scores) - 0.9) < 1e-9)

    m.CONFIG["segment_agg"] = "mean"
    check("mean", abs(m.aggregate_segment_scores(scores) - 0.4) < 1e-9)

    m.CONFIG["segment_agg"] = "topk_mean"
    m.CONFIG["segment_topk"] = 2
    check("topk_mean k=2", abs(m.aggregate_segment_scores(scores) - 0.65) < 1e-9)

    check("k > 길이면 전체 평균",
          abs(m.aggregate_segment_scores([0.2, 0.8]) - 0.5) < 1e-9)
    check("빈 입력 -> 0.0", m.aggregate_segment_scores([]) == 0.0)
    m.CONFIG["segment_agg"] = "max"


def test_fusion(m):
    print("\n[FILE_FAKE_PROB 융합]")
    vf, mf, vp, mp = 0.8, 0.3, 0.9, 0.4

    m.CONFIG["fusion_mode"] = "baseline"
    expected = max(vp * vf, mp * mf)
    check("baseline == max(VP*VF, MP*MF)",
          abs(m.combine_file_fake_score(vf, mf, vp, mp) - expected) < 1e-12)

    m.CONFIG["fusion_mode"] = "noisy_or"
    expected = 1 - (1 - vp * vf) * (1 - mp * mf)
    check("noisy_or 값", abs(m.combine_file_fake_score(vf, mf, vp, mp) - expected) < 1e-12)
    check("noisy_or >= baseline", m.combine_file_fake_score(vf, mf, vp, mp) >= max(vp * vf, mp * mf))

    m.CONFIG["fusion_mode"] = "gated_max"
    m.CONFIG["fusion_gate"] = 0.5
    check("gated_max: 음성만 게이트 통과 -> VF",
          abs(m.combine_file_fake_score(vf, mf, 0.9, 0.1) - vf) < 1e-12)
    check("gated_max: 둘 다 통과 -> max(VF, MF)",
          abs(m.combine_file_fake_score(vf, mf, 0.9, 0.9) - max(vf, mf)) < 1e-12)
    fallback = m.combine_file_fake_score(vf, mf, 0.1, 0.1)
    check("gated_max: 둘 다 미달 -> baseline 폴백",
          abs(fallback - max(0.1 * vf, 0.1 * mf)) < 1e-12, f"got {fallback}")

    m.CONFIG["fusion_mode"] = "gamma"
    m.CONFIG["fusion_gamma"] = 0.5
    expected = max((vp ** 0.5) * vf, (mp ** 0.5) * mf)
    check("gamma 값", abs(m.combine_file_fake_score(vf, mf, vp, mp) - expected) < 1e-12)

    print("\n[융합 출력 범위 — 무작위 10만 회]")
    rng = np.random.default_rng(0)
    samples = rng.random((100_000, 4))
    bad = {}
    for mode in ("baseline", "noisy_or", "gated_max", "gamma"):
        m.CONFIG["fusion_mode"] = mode
        worst_low, worst_high = 1.0, 0.0
        for a, b, c, d in samples[:20_000]:
            value = m.combine_file_fake_score(a, b, c, d)
            worst_low = min(worst_low, value)
            worst_high = max(worst_high, value)
        ok = 0.0 <= worst_low and worst_high <= 1.0
        check(f"{mode}: 출력이 [0,1]", ok, f"min={worst_low:.4f} max={worst_high:.4f}")
        if not ok:
            bad[mode] = (worst_low, worst_high)
    m.CONFIG["fusion_mode"] = "baseline"
    return bad


def test_rms(m):
    print("\n[RMS 무음 게이트]")
    check("빈 배열 -> 0.0", m.calculate_rms(np.zeros(0, dtype=np.float32)) == 0.0)
    check("무음 -> 게이트 미만", m.calculate_rms(np.zeros(1000, dtype=np.float32)) < m.SILENCE_RMS)
    check("일반 신호 -> 게이트 초과",
          m.calculate_rms(np.full(1000, 0.1, dtype=np.float32)) > m.SILENCE_RMS)


def test_file_matching(m):
    print("\n[입력 파일 매칭 — 베이스라인이 크래시하던 지점]")
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp)
        # 베이스라인 화이트리스트에 없는 확장자를 일부러 섞는다
        for name in ["TEST_0000.wav", "TEST_0001.mp3", "TEST_0002.flac",
                     "TEST_0003.amr", "TEST_0004.aiff", "TEST_0005.mp4"]:
            (test_dir / name).write_bytes(b"\x00")
        (test_dir / "sample_submission.csv").write_text("x", encoding="utf-8")
        (test_dir / "notes.txt").write_text("x", encoding="utf-8")

        mapping = m.build_id_to_path(test_dir)
        check("wav/mp3/flac 인식", all(f"TEST_000{i}" in mapping for i in (0, 1, 2)))
        check("amr/aiff/mp4 도 인식 (화이트리스트 제거 효과)",
              all(f"TEST_000{i}" in mapping for i in (3, 4, 5)),
              f"got {sorted(mapping)}")
        check("csv/txt 는 제외", "sample_submission" not in mapping and "notes" not in mapping)
        check("총 6개", len(mapping) == 6, f"got {len(mapping)}")


def test_sample_submission(m):
    print("\n[제출 양식 파싱]")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample_submission.csv"
        # BOM 포함 UTF-8 (데이콘 배포 파일에서 흔하다)
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ID"] + m.PREDICTION_COLUMNS)
            for i in range(3):
                writer.writerow([f"TEST_000{i}", 0, 0, 0, 0, 0])

        columns, rows = m.read_sample_submission(path)
        check("컬럼 순서 유지", columns == ["ID"] + m.PREDICTION_COLUMNS, str(columns))
        check("행 수", len(rows) == 3)
        check("BOM 제거된 ID", rows[0]["ID"] == "TEST_0000", repr(rows[0]["ID"]))

        bad = Path(tmp) / "bad.csv"
        bad.write_text("ID,FOO\nA,1\n", encoding="utf-8")
        try:
            m.read_sample_submission(bad)
            check("컬럼 누락 시 예외", False, "예외가 발생하지 않았다")
        except ValueError:
            check("컬럼 누락 시 예외", True)


def test_config_defaults(m):
    print("\n[기본 CONFIG = 베이스라인 동등성]")
    fresh = load_script()
    check("fusion_mode == baseline", fresh.CONFIG["fusion_mode"] == "baseline")
    check("segment_agg == max", fresh.CONFIG["segment_agg"] == "max")
    check("short_pad == tile", fresh.CONFIG["short_pad"] == "tile")
    check("max_segments == 0", fresh.CONFIG["max_segments"] == 0)
    check("SEGMENT_SAMPLES == 64600", fresh.SEGMENT_SAMPLES == 64_600)
    check("SILENCE_RMS == 1e-5", fresh.SILENCE_RMS == 1e-5)
    check("컬럼명 5개 확정", fresh.PREDICTION_COLUMNS == [
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"])


def main():
    print(f"대상: {SCRIPT}")
    module = load_script()

    test_segment_starts(module)
    test_segment_cap(module)
    test_padding(module)
    test_aggregation(module)
    test_fusion(module)
    test_rms(module)
    test_file_matching(module)
    test_sample_submission(module)
    test_config_defaults(module)
    test_gating_defaults(module)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    print("주의: GPU 경로(PANNs·HTDemucs·DF-Arena)는 여기서 검증되지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
