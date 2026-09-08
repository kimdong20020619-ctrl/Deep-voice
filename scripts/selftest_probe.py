#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""src/train/fit_probe.py 를 합성 임베딩으로 검증한다. GPU·실데이터 없이 돈다.

프로브는 Kaggle 에서 한 번 학습해 npz 로 굳히고 그대로 제출에 들어간다.
그 자리에서 처음 돌려보면 잘못을 알아채기 어렵다. 정답을 아는 인공 데이터로
"제대로 배우는가 · 새는 곳은 없는가 · script.py 와 계산이 같은가"를 먼저 확인한다.
"""

import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from train import fit_probe

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def make_cache(n_files=60, dim=32, segments=3, seed=0, separable=True):
    """정답을 아는 합성 캐시를 만든다.

    FAKE 파일의 임베딩만 특정 방향으로 밀어둔다. separable=False 면 신호가 없어
    프로브가 배울 것이 없어야 한다(과적합 탐지용 음성 대조군).
    """
    rng = np.random.default_rng(seed)
    direction = np.zeros(dim)
    direction[:4] = 1.0

    cache, rows = [], []
    for i in range(n_files):
        is_fake = i % 2
        base = rng.normal(size=(segments, dim))
        if separable and is_fake:
            base += 2.0 * direction
        audio_id = f"F{i:04d}"
        cache.append({
            "ID": audio_id,
            "music_embeddings": base,
            "original_embeddings": base,
            "voice_embeddings": None,
        })
        rows.append({
            "ID": audio_id,
            "FILE_FAKE_PROB": is_fake,
            "MUSIC_FAKE_PROB": is_fake,
            "MUSIC_PRESENT_PROB": 1,
        })
    return cache, rows


def main():
    print("=" * 60)
    print("[데이터셋 구성]")
    cache, rows = make_cache()
    X, y, groups = fit_probe.build_dataset(
        cache, rows, "music", "MUSIC_FAKE_PROB", "MUSIC_PRESENT_PROB")
    check("세그먼트 수 = 파일 x 세그먼트", X.shape == (60 * 3, 32), X.shape)
    check("라벨이 파일 라벨을 물려받는다", y.sum() == 30 * 3, y.sum())
    check("groups 가 파일 ID", len(np.unique(groups)) == 60, len(np.unique(groups)))

    print("\n[존재 마스크]")
    rows_absent = [dict(r, MUSIC_PRESENT_PROB=0) if i < 20 else r
                   for i, r in enumerate(rows)]
    Xa, _, _ = fit_probe.build_dataset(
        cache, rows_absent, "music", "MUSIC_FAKE_PROB", "MUSIC_PRESENT_PROB")
    check("음악 없는 파일은 제외된다", Xa.shape[0] == 40 * 3, Xa.shape)
    Xn, _, _ = fit_probe.build_dataset(cache, rows, "original", "FILE_FAKE_PROB")
    check("present_column=None 이면 전부 쓴다", Xn.shape[0] == 60 * 3, Xn.shape)

    print("\n[임베딩 없는 항목]")
    holey = [dict(c, music_embeddings=None) if i < 10 else c
             for i, c in enumerate(cache)]
    Xh, _, _ = fit_probe.build_dataset(
        holey, rows, "music", "MUSIC_FAKE_PROB", "MUSIC_PRESENT_PROB")
    check("임베딩 None 은 건너뛴다", Xh.shape[0] == 50 * 3, Xh.shape)

    print("\n[적합]")
    probe = fit_probe.fit(X, y)
    check("w 차원이 임베딩 차원과 같다", probe["w"].shape == (32,), probe["w"].shape)
    check("mean/scale 이 함께 나온다",
          probe["mean"].shape == (32,) and probe["scale"].shape == (32,))
    check("scale 에 0 이 없다", float(probe["scale"].min()) > 0, probe["scale"].min())

    scores = fit_probe.predict(probe, X)
    check("출력이 [0,1]", float(scores.min()) >= 0 and float(scores.max()) <= 1)
    check("FAKE 평균 > REAL 평균",
          scores[y == 1].mean() > scores[y == 0].mean(),
          f"{scores[y == 1].mean():.3f} vs {scores[y == 0].mean():.3f}")

    print("\n[분산 0 차원]")
    X_const = X.copy()
    X_const[:, 5] = 3.0
    probe_const = fit_probe.fit(X_const, y)
    check("상수 차원에서 NaN/inf 가 안 난다",
          np.isfinite(probe_const["w"]).all()
          and np.isfinite(fit_probe.predict(probe_const, X_const)).all())

    print("\n[교차검증 — 파일 단위 분할]")
    for train_mask, test_mask in fit_probe.grouped_folds(groups, n_folds=5, seed=0):
        overlap = set(groups[train_mask]) & set(groups[test_mask])
        if overlap:
            break
    else:
        overlap = set()
    check("같은 파일이 학습/검증에 동시에 안 들어간다", not overlap, sorted(overlap)[:3])

    cv = fit_probe.cross_validate(X, y, groups, n_folds=5)
    check("신호가 있으면 교차검증 AUC > 0.8", cv.mean() > 0.8, f"{cv.mean():.4f}")

    cache0, rows0 = make_cache(separable=False, seed=1)
    X0, y0, g0 = fit_probe.build_dataset(
        cache0, rows0, "music", "MUSIC_FAKE_PROB", "MUSIC_PRESENT_PROB")
    cv0 = fit_probe.cross_validate(X0, y0, g0, n_folds=5)
    check("신호가 없으면 교차검증 AUC ~ 0.5 (누수 탐지)",
          abs(cv0.mean() - 0.5) < 0.12, f"{cv0.mean():.4f}")

    print("\n[파일 단위 EER]")
    eer = fit_probe.file_level_eer(
        cache, rows, probe, "music", "MUSIC_FAKE_PROB",
        aggregate=max, present_column="MUSIC_PRESENT_PROB")
    check("분리 가능한 데이터에서 EER 이 낮다", eer < 0.15, f"{eer:.4f}")
    eer_sub = fit_probe.file_level_eer(
        cache, rows, probe, "music", "MUSIC_FAKE_PROB",
        aggregate=max, present_column="MUSIC_PRESENT_PROB",
        ids={c["ID"] for c in cache[:20]})
    check("ids 로 부분집합만 잰다", np.isfinite(eer_sub), eer_sub)

    print("\n[저장 형식 — script.py 의 LinearProbe 가 읽는 형식]")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = fit_probe.save(Path(tmp) / "music_head.npz", probe)
        # np.load 는 파일 핸들을 열어둔다. 윈도에서는 그 상태로 지우면 PermissionError 다.
        with np.load(path) as data:
            files = set(data.files)
            w = np.asarray(data["w"], dtype=np.float64).reshape(-1)
            b = float(np.asarray(data["b"]).reshape(-1)[0])
            mean = np.asarray(data["mean"], dtype=np.float64).reshape(-1)
            scale = np.asarray(data["scale"], dtype=np.float64).reshape(-1)
        check("키가 w/b/mean/scale", files == {"w", "b", "mean", "scale"}, files)

        # script.py 의 계산을 그대로 재현해 값이 같은지 본다.
        z = (X - mean) / scale
        replayed = 1.0 / (1.0 + np.exp(-np.clip(z @ w + b, -60.0, 60.0)))
        check("script.py 와 계산이 일치한다",
              np.allclose(replayed, scores, atol=1e-12),
              float(np.abs(replayed - scores).max()))

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"실패 {len(FAILURES)}건: {FAILURES}")
        return 1
    print("전체 통과")
    print("주의: 합성 임베딩이다. 실제 DF-Arena 임베딩의 분리도는 여기서 알 수 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
