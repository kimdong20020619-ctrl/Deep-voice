#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DF-Arena 임베딩 위에 얹는 선형 프로브를 적합한다.

DF-Arena 1B 는 ASVspoof 계열 **음성** 위조 탐지기다. 생성 음악을 학습한 적이 없는데
대회 실효 가중치는 MUSIC 0.27 · FILE 0.45 다. 1B 모델을 파인튜닝하는 대신
마지막 분류기(fc5) 직전 임베딩(1280차원)에 선형 하나를 얹는다.

  새 의존성 0 · 새 대용량 가중치 0(수 KB npz) · 추론 시간 0
  (임베딩은 추론 중 어차피 계산된다)

출력 npz 형식은 submit/script.py 의 LinearProbe 가 그대로 읽는다:
    w (dim,) · b scalar · mean (dim,) · scale (dim,)

주의 — 학습 임베딩은 **추론과 같은 경로**로 뽑아야 한다.
src/eval/sweep.py 의 precompute(want_embeddings=True) 가 그 경로다.
혼합 파일은 Demucs 스템, 음악만 있는 파일은 게이팅으로 원본이 들어간다.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

# 레포 관례대로 src/ 를 sys.path 에 올린 상태를 전제한다
# (노트북도 sys.path.insert(0, ".../submit/src") 후 from eval import sweep 로 쓴다).
from eval.metric import compute_eer

# precompute 가 만드는 스템 이름. 프로브를 붙일 대상이다.
STEM_MUSIC = "music"
STEM_ORIGINAL = "original"


def build_dataset(cache, label_rows, stem, label_column, present_column=None):
    """세그먼트 단위 학습셋을 만든다.

    반환 (X, y, groups) — groups 는 파일 ID 다. 같은 파일의 세그먼트가
    학습·검증으로 갈라지면 성능이 부풀려지므로 분할은 항상 이 단위로 한다.
    """
    index = {row["ID"]: row for row in label_rows}
    features, labels, groups = [], [], []

    for entry in cache:
        row = index.get(entry["ID"])
        if row is None:
            continue
        # Music EER 이 음악 존재 샘플에서만 계산되듯, 프로브도 같은 모집단에서 배운다.
        if present_column is not None and int(row[present_column]) != 1:
            continue
        embeddings = entry.get(f"{stem}_embeddings")
        if embeddings is None or len(embeddings) == 0:
            continue
        embeddings = np.asarray(embeddings, dtype=np.float64)
        features.append(embeddings)
        labels.append(np.full(embeddings.shape[0], int(row[label_column])))
        groups.append(np.full(embeddings.shape[0], entry["ID"]))

    assert features, ("임베딩이 하나도 없다 — "
                      f"precompute(want_embeddings=True) 로 뽑았는가? (stem={stem})")
    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    g = np.concatenate(groups, axis=0)
    assert len(np.unique(y)) == 2, f"한쪽 클래스만 있다: {np.unique(y)} — 검증셋 구성이 잘못됐다"
    return X, y, g


def _standardize(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    # 분산이 0 인 차원은 나누면 inf 가 된다. 1 로 두면 (x-mean)/1 = 0 이라 무해하다.
    scale[scale < 1e-8] = 1.0
    return mean, scale


def fit(X, y, C=1.0, seed=0):
    """L2 로지스틱 회귀. 클래스 불균형은 가중치로 잡는다."""
    mean, scale = _standardize(X)
    Z = (X - mean) / scale

    # penalty="l2" 를 명시하지 않는다. sklearn 1.8 에서 폐기됐고 1.10 에서 사라지는데,
    # 기본값이 어느 버전에서나 L2 라 생략하는 쪽이 이식성이 높다.
    # Kaggle 과 로컬의 sklearn 버전이 다를 수 있다.
    model = LogisticRegression(
        C=C, class_weight="balanced",
        max_iter=2000, random_state=seed,
    )
    model.fit(Z, y)

    return {
        "w": model.coef_.reshape(-1).astype(np.float64),
        "b": float(model.intercept_.reshape(-1)[0]),
        "mean": mean.astype(np.float64),
        "scale": scale.astype(np.float64),
    }


def predict(probe, embeddings):
    """submit/script.py 의 LinearProbe.predict 와 같은 계산."""
    z = (np.asarray(embeddings, dtype=np.float64) - probe["mean"]) / probe["scale"]
    logit = z @ probe["w"] + probe["b"]
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60.0, 60.0)))


def file_level_eer(cache, label_rows, probe, stem, label_column,
                   aggregate, present_column=None, ids=None):
    """세그먼트 점수를 파일 점수로 집계한 뒤 EER 을 잰다.

    세그먼트 AUC 가 아니라 이 값이 리더보드와 같은 단위다.
    aggregate 는 점수 리스트를 float 하나로 줄이는 함수다
    (submit/script.py 의 aggregate_segment_scores 를 그대로 넘긴다).
    """
    index = {row["ID"]: row for row in label_rows}
    truth, scores = [], []
    for entry in cache:
        if ids is not None and entry["ID"] not in ids:
            continue
        row = index.get(entry["ID"])
        if row is None:
            continue
        if present_column is not None and int(row[present_column]) != 1:
            continue
        embeddings = entry.get(f"{stem}_embeddings")
        if embeddings is None or len(embeddings) == 0:
            continue
        truth.append(int(row[label_column]))
        scores.append(float(aggregate(predict(probe, embeddings).tolist())))
    if len(truth) < 2:
        return float("nan")
    return compute_eer(np.asarray(truth), np.asarray(scores))


def grouped_folds(groups, n_folds=5, seed=0):
    """파일 단위로 나눈 교차검증 폴드. 같은 파일이 양쪽에 걸치지 않는다."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    chunks = np.array_split(unique, n_folds)
    for held in chunks:
        held_set = set(held.tolist())
        mask = np.array([g in held_set for g in groups])
        yield ~mask, mask


def cross_validate(X, y, groups, C=1.0, n_folds=5, seed=0):
    """폴드별 세그먼트 AUC. 과적합 여부를 싸게 보는 용도다."""
    from sklearn.metrics import roc_auc_score

    scores = []
    for train_mask, test_mask in grouped_folds(groups, n_folds, seed):
        if len(np.unique(y[train_mask])) < 2 or len(np.unique(y[test_mask])) < 2:
            continue
        probe = fit(X[train_mask], y[train_mask], C=C, seed=seed)
        scores.append(roc_auc_score(y[test_mask], predict(probe, X[test_mask])))
    return np.asarray(scores)


def save(path, probe):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, w=probe["w"], b=probe["b"],
             mean=probe["mean"], scale=probe["scale"])
    return path


def describe(probe, X=None, y=None, cv_scores=None):
    lines = [f"차원 {probe['w'].size} · |w| {np.linalg.norm(probe['w']):.4f} · b {probe['b']:+.4f}"]
    if X is not None and y is not None:
        from sklearn.metrics import roc_auc_score
        lines.append(f"학습셋 AUC {roc_auc_score(y, predict(probe, X)):.4f} "
                     f"(n={len(y)}, FAKE {int(y.sum())})")
    if cv_scores is not None and len(cv_scores):
        lines.append(f"교차검증 AUC {cv_scores.mean():.4f} ± {cv_scores.std():.4f} "
                     f"({len(cv_scores)}폴드, 파일 단위 분할)")
    return "\n".join(lines)
