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

# script.py 를 import 하면 submit/__pycache__ 가 생기고, 그게 제출 zip 에 섞이면
# 규격 외 최상위 항목이 되어 설치 오류가 난다. 아예 만들지 않는다.
sys.dont_write_bytecode = True

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

    # 비율 지정 top-k — max 의 길이 편향을 상쇄하는 모드
    m.CONFIG["segment_topk_ratio"] = 0.25
    ten = [0.1 * i for i in range(1, 11)]          # 0.1 .. 1.0, 10개
    check("비율 25% -> 상위 3개 평균 (ceil(10*0.25)=3)",
          abs(m.aggregate_segment_scores(ten) - (0.8 + 0.9 + 1.0) / 3) < 1e-9,
          str(m.aggregate_segment_scores(ten)))
    four = [0.2, 0.4, 0.6, 0.8]
    check("비율이 k<1 이 되면 최소 1개",
          abs(m.aggregate_segment_scores(four) - 0.8) < 1e-9,
          str(m.aggregate_segment_scores(four)))
    check("세그먼트 1개면 그 값", abs(m.aggregate_segment_scores([0.42]) - 0.42) < 1e-9)

    # 길이 편향이 실제로 줄어드는가 — 같은 분포에서 길이만 다르게
    import numpy as _np
    rng = _np.random.default_rng(0)
    m.CONFIG["segment_agg"] = "max"
    short_max = _np.mean([m.aggregate_segment_scores(rng.random(2)) for _ in range(2000)])
    long_max = _np.mean([m.aggregate_segment_scores(rng.random(15)) for _ in range(2000)])
    m.CONFIG["segment_agg"] = "topk_mean"
    m.CONFIG["segment_topk_ratio"] = 0.25
    short_r = _np.mean([m.aggregate_segment_scores(rng.random(2)) for _ in range(2000)])
    long_r = _np.mean([m.aggregate_segment_scores(rng.random(15)) for _ in range(2000)])
    check("max 는 길이에 따라 점수가 오른다 (편향 존재)", long_max - short_max > 0.15,
          f"2seg {short_max:.3f} -> 15seg {long_max:.3f}")
    check("비율 top-k 는 편향이 작다",
          abs(long_r - short_r) < abs(long_max - short_max),
          f"비율 {abs(long_r - short_r):.3f} vs max {abs(long_max - short_max):.3f}")

    m.CONFIG["segment_topk_ratio"] = 0.0
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
        ok = worst_low >= 0.0 and worst_high <= 1.0
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


class FakeScorer:
    """DF-Arena 대역. kind 로 어떤 입력이 왔는지 구분하고 호출 횟수를 센다."""

    def __init__(self, voice, music, direct):
        self.values = {"voice": voice, "music": music, None: direct}
        self.calls = []

    def score(self, audio, kind=None, want_embeddings=False):
        self.calls.append(kind)
        value = 0.0 if audio.size <= 1 else self.values[kind]
        if want_embeddings:
            embeddings = None if audio.size <= 1 else np.ones((2, 4), dtype=np.float32)
            return value, embeddings
        return value


def run_pipeline(m, voice_present, music_present, voice=0.8, music=0.3, direct=0.6):
    """process_one_file 을 GPU 없이 돌린다. 무거운 단계만 갈아끼운다."""
    audio = np.full(64_000, 0.1, dtype=np.float32)
    m.load_audio_16k = lambda path: audio
    m.predict_presence = lambda panns, wav: (voice_present, music_present)
    m.separate_voice_and_music = lambda wav, model, device: (
        np.full(64_000, 0.1, dtype=np.float32), np.full(64_000, 0.1, dtype=np.float32))
    scorer = FakeScorer(voice, music, direct)
    result = m.process_one_file(Path("x.wav"), None, scorer, None, None)
    return result, scorer


def test_file_head(m):
    print("\n[FILE_FAKE 헤드 — 실효 가중치 0.45]")
    m.CONFIG["demucs_gating"] = True
    m.CONFIG["gate_voice"] = 0.20
    m.CONFIG["gate_music"] = 0.05
    vf, mf, direct = 0.8, 0.3, 0.6

    # --- 혼합 파일 (분리 수행) ---
    m.CONFIG["file_head"] = "fusion"
    res, sc = run_pipeline(m, 0.9, 0.9, vf, mf, direct)
    fused = max(0.9 * vf, 0.9 * mf)
    check("fusion: 베이스라인 값 유지", abs(res["FILE_FAKE_PROB"] - fused) < 1e-12,
          f"got {res['FILE_FAKE_PROB']}")
    check("fusion: DF-Arena 2회만 호출", sc.calls == ["voice", "music"], str(sc.calls))

    m.CONFIG["file_head"] = "direct"
    res, sc = run_pipeline(m, 0.9, 0.9, vf, mf, direct)
    check("direct: 원본 점수", abs(res["FILE_FAKE_PROB"] - direct) < 1e-12,
          f"got {res['FILE_FAKE_PROB']}")
    check("direct: 혼합 파일은 3회 호출", sc.calls == ["voice", "music", None], str(sc.calls))

    m.CONFIG["file_head"] = "direct_max"
    res, _ = run_pipeline(m, 0.9, 0.9, vf, mf, direct)
    check("direct_max = max(direct, fusion)",
          abs(res["FILE_FAKE_PROB"] - max(direct, fused)) < 1e-12, f"got {res['FILE_FAKE_PROB']}")

    m.CONFIG["file_head"] = "direct_mean"
    res, _ = run_pipeline(m, 0.9, 0.9, vf, mf, direct)
    check("direct_mean = 평균",
          abs(res["FILE_FAKE_PROB"] - 0.5 * (direct + fused)) < 1e-12,
          f"got {res['FILE_FAKE_PROB']}")

    # --- 음성 단독 (게이팅으로 분리 생략) — 원본 재사용으로 추가 비용이 0이어야 한다 ---
    m.CONFIG["file_head"] = "direct"
    res, sc = run_pipeline(m, 0.9, 0.01, vf, mf, direct)
    check("음성 단독: 분리 생략 시 원본 재사용 (추가 호출 없음)",
          sc.calls == ["voice", "music"], str(sc.calls))
    check("음성 단독: FILE == 음성 점수",
          abs(res["FILE_FAKE_PROB"] - vf) < 1e-12, f"got {res['FILE_FAKE_PROB']}")
    check("음성 단독: MUSIC_FAKE == 0", res["MUSIC_FAKE_PROB"] == 0.0)

    # --- 음악 단독 ---
    res, sc = run_pipeline(m, 0.01, 0.9, vf, mf, direct)
    check("음악 단독: 추가 호출 없음", sc.calls == ["voice", "music"], str(sc.calls))
    check("음악 단독: FILE == 음악 점수",
          abs(res["FILE_FAKE_PROB"] - mf) < 1e-12, f"got {res['FILE_FAKE_PROB']}")
    check("음악 단독: VOICE_FAKE == 0", res["VOICE_FAKE_PROB"] == 0.0)

    # --- 집계 덮어쓰기가 걸리면 재사용하지 않아야 한다 ---
    m.CONFIG["segment_agg_voice"] = "mean"
    res, sc = run_pipeline(m, 0.9, 0.01, vf, mf, direct)
    check("집계 덮어쓰기 시 재사용 금지 (원본 재계산)",
          sc.calls == ["voice", "music", None], str(sc.calls))
    m.CONFIG["segment_agg_voice"] = None

    m.CONFIG["file_head"] = "fusion"
    m.CONFIG["demucs_gating"] = True


def test_head_aggregation(m):
    print("\n[헤드별 집계 덮어쓰기]")
    m.CONFIG["segment_agg"] = "max"
    m.CONFIG["segment_agg_voice"] = None
    m.CONFIG["segment_agg_music"] = None
    check("덮어쓰기 없음 -> 전역값", m.resolve_agg("music") == "max")
    check("kind 미지정 -> 전역값", m.resolve_agg() == "max")

    m.CONFIG["segment_agg_music"] = "mean"
    check("음악만 mean", m.resolve_agg("music") == "mean")
    check("음성은 전역 유지", m.resolve_agg("voice") == "max")

    scores = [0.1, 0.9, 0.4, 0.2]
    check("명시 모드가 CONFIG 를 이긴다",
          abs(m.aggregate_segment_scores(scores, "mean") - 0.4) < 1e-9)
    check("모드 미지정이면 CONFIG",
          abs(m.aggregate_segment_scores(scores) - 0.9) < 1e-9)
    m.CONFIG["segment_agg_music"] = None


def test_music_probe(m):
    print("\n[생성 음악 선형 프로브 — 실효 가중치 0.27]")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "music_head.npz"

        # w=[1,0,0,0], b=0 -> 첫 차원만 보는 프로브
        np.savez(path, w=np.array([1.0, 0.0, 0.0, 0.0]), b=np.array([0.0]))
        probe = m.MusicProbe(path)
        out = probe.predict(np.array([[0.0, 9, 9, 9], [2.0, 0, 0, 0], [-2.0, 0, 0, 0]]))
        check("logit 0 -> 0.5", abs(out[0] - 0.5) < 1e-12, str(out[0]))
        check("양수 logit -> >0.5", out[1] > 0.5 and abs(out[1] - 1 / (1 + np.exp(-2))) < 1e-12)
        check("음수 logit -> <0.5", out[2] < 0.5)
        check("출력이 [0,1]", bool(((out >= 0) & (out <= 1)).all()))

        # 극단 logit 에서 오버플로가 나지 않아야 한다
        big = probe.predict(np.array([[1e6, 0, 0, 0], [-1e6, 0, 0, 0]]))
        check("극단값에서 유한", bool(np.isfinite(big).all()), str(big))

        # 표준화 필드 반영
        np.savez(path, w=np.array([1.0, 0, 0, 0]), b=np.array([0.0]),
                 mean=np.array([2.0, 0, 0, 0]), scale=np.array([2.0, 1, 1, 1]))
        probe2 = m.MusicProbe(path)
        got = probe2.predict(np.array([[6.0, 0, 0, 0]]))[0]
        check("mean/scale 적용 ((6-2)/2=2)",
              abs(got - 1 / (1 + np.exp(-2))) < 1e-12, str(got))

    print("\n[프로브 통합 — 블렌드]")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "music_head.npz"
        np.savez(path, w=np.zeros(4), b=np.array([0.0]))   # 항상 0.5 를 내는 프로브
        probe = m.MusicProbe(path)

        m.CONFIG["file_head"] = "fusion"
        m.CONFIG["demucs_gating"] = True
        audio = np.full(64_000, 0.1, dtype=np.float32)
        m.load_audio_16k = lambda p: audio
        m.predict_presence = lambda panns, wav: (0.9, 0.9)
        m.separate_voice_and_music = lambda wav, model, device: (audio, audio)

        scorer = FakeScorer(0.8, 0.3, 0.6)
        m.CONFIG["music_head_blend"] = 1.0
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, probe)
        check("blend=1.0 -> 프로브 값만", abs(res["MUSIC_FAKE_PROB"] - 0.5) < 1e-9,
              str(res["MUSIC_FAKE_PROB"]))

        scorer = FakeScorer(0.8, 0.3, 0.6)
        m.CONFIG["music_head_blend"] = 0.5
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, probe)
        check("blend=0.5 -> 프로브와 DF-Arena 평균",
              abs(res["MUSIC_FAKE_PROB"] - 0.5 * (0.5 + 0.3)) < 1e-9,
              str(res["MUSIC_FAKE_PROB"]))

        scorer = FakeScorer(0.8, 0.3, 0.6)
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, None)
        check("프로브 None -> 기존 동작 유지",
              abs(res["MUSIC_FAKE_PROB"] - 0.3) < 1e-9, str(res["MUSIC_FAKE_PROB"]))
        m.CONFIG["music_head_blend"] = 1.0

    print("\n[FILE 프로브 — 실효 가중치 0.45]")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "file_head.npz"
        np.savez(path, w=np.zeros(4), b=np.array([0.0]))   # 항상 0.5 를 내는 프로브
        file_probe = m.LinearProbe(path)

        m.CONFIG["file_head"] = "direct"
        m.CONFIG["demucs_gating"] = True
        audio = np.full(64_000, 0.1, dtype=np.float32)
        m.load_audio_16k = lambda p: audio
        m.separate_voice_and_music = lambda wav, model, device: (audio, audio)

        # 혼합 파일 — 분리가 일어나고 원본을 따로 채점한다
        m.predict_presence = lambda panns, wav: (0.9, 0.9)
        scorer = FakeScorer(0.8, 0.3, 0.6)
        m.CONFIG["file_probe_blend"] = 1.0
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, None, file_probe)
        check("blend=1.0 -> FILE 이 프로브 값", abs(res["FILE_FAKE_PROB"] - 0.5) < 1e-9,
              str(res["FILE_FAKE_PROB"]))

        scorer = FakeScorer(0.8, 0.3, 0.6)
        m.CONFIG["file_probe_blend"] = 0.5
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, None, file_probe)
        check("blend=0.5 -> 프로브와 direct 평균",
              abs(res["FILE_FAKE_PROB"] - 0.5 * (0.5 + 0.6)) < 1e-9,
              str(res["FILE_FAKE_PROB"]))
        m.CONFIG["file_probe_blend"] = 1.0

        scorer = FakeScorer(0.8, 0.3, 0.6)
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, None, None)
        check("프로브 None -> 기존 direct 유지",
              abs(res["FILE_FAKE_PROB"] - 0.6) < 1e-9, str(res["FILE_FAKE_PROB"]))

        # 게이팅으로 분리를 건너뛴 파일 — 원본이 곧 음악 스템이다.
        # 여기서 DF-Arena 를 한 번 더 부르면 추론 시간이 늘어난다. 임베딩을 재사용해야 한다.
        m.predict_presence = lambda panns, wav: (0.01, 0.9)
        scorer = FakeScorer(0.8, 0.3, 0.6)
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, None, file_probe)
        check("게이팅 시 원본 재채점 없음 (임베딩 재사용)",
              None not in scorer.calls, str(scorer.calls))
        check("게이팅 시에도 프로브가 적용된다",
              abs(res["FILE_FAKE_PROB"] - 0.5) < 1e-9, str(res["FILE_FAKE_PROB"]))

        # 음악 프로브가 켜져 있어도 direct 는 **프로브 이전 원점수**를 써야 한다.
        # 프로브가 섞인 값을 재사용하면 direct 의 의미가 조용히 바뀐다.
        music_path = Path(tmp) / "music_head.npz"
        np.savez(music_path, w=np.zeros(4), b=np.array([0.0]))
        music_probe = m.LinearProbe(music_path)
        scorer = FakeScorer(0.8, 0.3, 0.6)
        res = m.process_one_file(Path("x.wav"), None, scorer, None, None, music_probe, None)
        check("음악 프로브가 direct 를 오염시키지 않는다",
              abs(res["FILE_FAKE_PROB"] - 0.3) < 1e-9, str(res["FILE_FAKE_PROB"]))
        check("그때 MUSIC 은 프로브 값", abs(res["MUSIC_FAKE_PROB"] - 0.5) < 1e-9,
              str(res["MUSIC_FAKE_PROB"]))

        # 로더 — 파일이 없거나 fusion 이면 조용히 비활성
        m.CONFIG["file_probe"] = "probe"
        m.CONFIG["file_head"] = "fusion"
        check("file_head=fusion 이면 FILE 프로브를 안 켠다", m.load_file_probe() is None)
        m.CONFIG["file_head"] = "direct"
        check("npz 가 없으면 조용히 비활성", m.load_file_probe() is None)
        m.CONFIG["file_probe"] = "none"
        check("file_probe=none 이면 비활성", m.load_file_probe() is None)

    # 게이팅 상태를 되돌린다. 안 그러면 다음 섹션이 음악 단독 파일을 보게 된다.
    m.predict_presence = lambda panns, wav: (0.9, 0.9)

    print("\n[진단 모드 — Music EER 단독 측정]")
    m.CONFIG["file_head"] = "direct"
    m.CONFIG["music_head"] = "constant"
    m.CONFIG["music_constant"] = 0.5
    scorer = FakeScorer(0.8, 0.3, 0.6)
    res = m.process_one_file(Path("x.wav"), None, scorer, None, None, None)
    check("MUSIC 이 상수로 고정", res["MUSIC_FAKE_PROB"] == 0.5, str(res["MUSIC_FAKE_PROB"]))
    check("FILE 은 direct 유지 (음악에 비의존)",
          abs(res["FILE_FAKE_PROB"] - 0.6) < 1e-9, str(res["FILE_FAKE_PROB"]))
    check("VOICE 는 그대로", abs(res["VOICE_FAKE_PROB"] - 0.8) < 1e-9)
    m.CONFIG["music_head"] = "none"

    print("\n[진단 모드 — Voice EER 단독 측정]")
    m.CONFIG["voice_head"] = "constant"
    m.CONFIG["voice_constant"] = 0.5
    scorer = FakeScorer(0.8, 0.3, 0.6)
    res = m.process_one_file(Path("x.wav"), None, scorer, None, None)
    check("VOICE 가 상수로 고정", res["VOICE_FAKE_PROB"] == 0.5, str(res["VOICE_FAKE_PROB"]))
    check("FILE 은 direct 유지 (음성에 비의존)",
          abs(res["FILE_FAKE_PROB"] - 0.6) < 1e-9, str(res["FILE_FAKE_PROB"]))
    check("MUSIC 은 그대로", abs(res["MUSIC_FAKE_PROB"] - 0.3) < 1e-9,
          str(res["MUSIC_FAKE_PROB"]))

    # 게이팅으로 분리를 건너뛴 음성 단독 파일 — direct 가 voice_fake 를 재사용한다.
    # 상수 덮어쓰기가 그보다 먼저 일어나면 FILE 까지 0.5 가 되어 역산이 무효가 된다.
    m.predict_presence = lambda panns, wav: (0.9, 0.01)
    scorer = FakeScorer(0.8, 0.3, 0.6)
    res = m.process_one_file(Path("x.wav"), None, scorer, None, None)
    check("음성 단독 게이팅에서도 FILE 이 원점수(0.8)",
          abs(res["FILE_FAKE_PROB"] - 0.8) < 1e-9, str(res["FILE_FAKE_PROB"]))
    check("그때 VOICE 는 상수", res["VOICE_FAKE_PROB"] == 0.5, str(res["VOICE_FAKE_PROB"]))
    m.predict_presence = lambda panns, wav: (0.9, 0.9)

    m.CONFIG["voice_head"] = "none"
    m.CONFIG["file_head"] = "fusion"


def test_config_defaults(m):
    print("\n[기본 CONFIG = 베이스라인 동등성]")
    fresh = load_script()
    check("fusion_mode == baseline", fresh.CONFIG["fusion_mode"] == "baseline")
    check("segment_agg == max", fresh.CONFIG["segment_agg"] == "max")
    check("short_pad == tile", fresh.CONFIG["short_pad"] == "tile")
    check("max_segments == 0", fresh.CONFIG["max_segments"] == 0)
    check("file_head == direct (제출 #2)", fresh.CONFIG["file_head"] == "direct")
    check("music_head == none (가중치 미학습)", fresh.CONFIG["music_head"] == "none")
    check("헤드별 집계 덮어쓰기 없음",
          fresh.CONFIG["segment_agg_voice"] is None and fresh.CONFIG["segment_agg_music"] is None)
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
    test_head_aggregation(module)
    test_file_head(module)
    test_music_probe(module)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    print("주의: GPU 경로(PANNs·HTDemucs·DF-Arena)는 여기서 검증되지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
