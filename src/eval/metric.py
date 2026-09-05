#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""대회 평가 산식을 그대로 구현한다.

이 파일이 로컬 검증의 측정 도구다. 리더보드 제출은 하루 3회뿐이므로
설정 비교는 전부 여기서 하고 이긴 것만 제출한다.

산식 (대회 평가 탭 원문):

    Score = 0.9 x ADS + 0.1 x CPS
    ADS   = 0.5 x (1 - File EER) + 0.2 x (1 - Voice EER) + 0.3 x (1 - Music EER)
    CPS   = 0.5 x Voice Presence ROC-AUC + 0.5 x Music Presence ROC-AUC

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    EER = (fpr[idx] + fnr[idx]) / 2

- FAKE 가 양성 클래스(1)
- Voice EER 은 음성이 존재하는 샘플에서만, Music EER 은 음악이 존재하는 샘플에서만 계산
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# 총점 가중치
SCORE_ADS_WEIGHT = 0.9
SCORE_CPS_WEIGHT = 0.1

# ADS 내부 가중치
ADS_FILE_WEIGHT = 0.5
ADS_VOICE_WEIGHT = 0.2
ADS_MUSIC_WEIGHT = 0.3

# 총점 기준 실효 가중치 — 자원 배분의 근거
EFFECTIVE_WEIGHTS = {
    "file": SCORE_ADS_WEIGHT * ADS_FILE_WEIGHT,      # 0.45
    "music": SCORE_ADS_WEIGHT * ADS_MUSIC_WEIGHT,    # 0.27
    "voice": SCORE_ADS_WEIGHT * ADS_VOICE_WEIGHT,    # 0.18
    "presence": SCORE_CPS_WEIGHT,                    # 0.10 (둘 합쳐)
}


def compute_eer(y_true, y_score):
    """대회가 제시한 코드 그대로. 양성 클래스는 1(FAKE)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)

    if y_true.size == 0:
        return float("nan")
    # 한쪽 클래스만 있으면 ROC 가 정의되지 않는다. 검증셋 구성 오류를 조용히 넘기지 않는다.
    if len(np.unique(y_true)) < 2:
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def evaluate(predictions, truth):
    """예측과 정답으로 대회 점수를 계산한다.

    predictions, truth 는 각각 다음 키를 갖는 딕셔너리 (값은 길이 N 배열):
        FILE_FAKE_PROB, VOICE_FAKE_PROB, MUSIC_FAKE_PROB,
        VOICE_PRESENT_PROB, MUSIC_PRESENT_PROB
    truth 는 0/1 정수여야 한다.

    반환: 점수와 하위 지표를 담은 딕셔너리.
    """
    file_fake = np.asarray(truth["FILE_FAKE_PROB"]).astype(int)
    voice_fake = np.asarray(truth["VOICE_FAKE_PROB"]).astype(int)
    music_fake = np.asarray(truth["MUSIC_FAKE_PROB"]).astype(int)
    voice_present = np.asarray(truth["VOICE_PRESENT_PROB"]).astype(int)
    music_present = np.asarray(truth["MUSIC_PRESENT_PROB"]).astype(int)

    # Voice EER 은 음성이 존재하는 샘플에서만, Music EER 은 음악이 존재하는 샘플에서만.
    voice_mask = voice_present == 1
    music_mask = music_present == 1

    file_eer = compute_eer(file_fake, predictions["FILE_FAKE_PROB"])
    voice_eer = compute_eer(voice_fake[voice_mask],
                            np.asarray(predictions["VOICE_FAKE_PROB"])[voice_mask])
    music_eer = compute_eer(music_fake[music_mask],
                            np.asarray(predictions["MUSIC_FAKE_PROB"])[music_mask])

    ads = (ADS_FILE_WEIGHT * (1 - file_eer)
           + ADS_VOICE_WEIGHT * (1 - voice_eer)
           + ADS_MUSIC_WEIGHT * (1 - music_eer))

    voice_auc = roc_auc_score(voice_present, predictions["VOICE_PRESENT_PROB"])
    music_auc = roc_auc_score(music_present, predictions["MUSIC_PRESENT_PROB"])
    cps = 0.5 * voice_auc + 0.5 * music_auc

    return {
        "score": SCORE_ADS_WEIGHT * ads + SCORE_CPS_WEIGHT * cps,
        "ads": float(ads),
        "cps": float(cps),
        "file_eer": file_eer,
        "voice_eer": voice_eer,
        "music_eer": music_eer,
        "voice_auc": float(voice_auc),
        "music_auc": float(music_auc),
        "n": int(file_fake.size),
        "n_voice": int(voice_mask.sum()),
        "n_music": int(music_mask.sum()),
    }


def format_result(result, label=""):
    return (
        f"{label:24s} Score {result['score']:.5f} | ADS {result['ads']:.5f} "
        f"| CPS {result['cps']:.5f} || EER  file {result['file_eer']:.4f} "
        f"voice {result['voice_eer']:.4f} music {result['music_eer']:.4f}"
    )


def score_delta_explained(before, after):
    """두 결과의 총점 차이를 항목별 기여로 분해한다.

    어떤 변경이 어디를 움직였는지 보려면 이 분해가 필요하다.
    """
    parts = {
        "file": SCORE_ADS_WEIGHT * ADS_FILE_WEIGHT * (before["file_eer"] - after["file_eer"]),
        "voice": SCORE_ADS_WEIGHT * ADS_VOICE_WEIGHT * (before["voice_eer"] - after["voice_eer"]),
        "music": SCORE_ADS_WEIGHT * ADS_MUSIC_WEIGHT * (before["music_eer"] - after["music_eer"]),
        "presence": SCORE_CPS_WEIGHT * (after["cps"] - before["cps"]),
    }
    parts["total"] = sum(parts.values())
    return parts


if __name__ == "__main__":
    # 리더보드 1위 값을 역산해 산식이 맞는지 확인한다.
    ads, cps = 0.82323, 0.98932
    print(f"1위 재현: {SCORE_ADS_WEIGHT * ads + SCORE_CPS_WEIGHT * cps:.5f}  (실제 0.83984)")
    ads, cps = 0.6597777778, 0.9893940741
    print(f"제출#1 재현: {SCORE_ADS_WEIGHT * ads + SCORE_CPS_WEIGHT * cps:.10f}  "
          f"(실제 0.6927394074)")
