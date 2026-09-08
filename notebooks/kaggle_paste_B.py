# ===== B 블록 : 2단계 — 설정 비교 (몇 초, 몇 번이든 다시 실행 가능) =====
# Kaggle 새 노트북에 이 전체를 붙여넣고 실행한다.
# Import Notebook 이 안 될 때 쓰는 우회로이며 내용은 노트북과 동일하다.


# ---------- B1 ----------
results = sweep_mod.sweep(script, cache, labels, sweep_mod.default_grid(),
                          baseline_name="baseline(fusion)")

# ---------- B2 ----------
best = results[0]
print("1단계 1위:", best["name"], best["config"], f"Score {best['score']:.5f}\n")

extra = [
    ("+agg:topk2", {"segment_agg": "topk_mean", "segment_topk": 2}),
    ("+agg:mean", {"segment_agg": "mean"}),
    ("+music_agg:mean", {"segment_agg_music": "mean"}),
    ("+gate:m0.02", {"gate_music": 0.02}),
    ("+gate:m0.20", {"gate_music": 0.20}),
    ("+gate:v0.40", {"gate_voice": 0.40}),
    ("+gate:off", {"demucs_gating": False}),
]
stage2 = [(best["name"], best["config"])] + sweep_mod.cross_grid(best["config"], extra)
results2 = sweep_mod.sweep(script, cache, labels, stage2, baseline_name=best["name"])

# ---------- B3 ----------
from eval.metric import rank_score

print("최종 순위")
for rank, r in enumerate(results2, 1):
    # Music EER 이 nan 이면 공식 총점도 nan 이다. rank_score 가 측정 가능한 항목만 재정규화한다.
    print(f"{rank:>2}. {r['name']:<28} Score {rank_score(r):.5f}  "
          f"(file {r['file_eer']:.4f} / voice {r['voice_eer']:.4f} / music {r['music_eer']:.4f})")
print("\n최고 설정 CONFIG 덮어쓰기:")
print(results2[0]["config"])

# ---------- B4 ----------
import numpy as np
from train import fit_probe

MUSIC_HEAD = "/kaggle/working/submit/model/music_head.npz"
FILE_HEAD = "/kaggle/working/submit/model/file_head.npz"

has_fake_music = any(int(r["MUSIC_FAKE_PROB"]) for r in labels)
print("가짜 음악:", "있음" if has_fake_music else "없음 -> 음악 프로브 학습 불가")

music_probe = file_probe = None

if has_fake_music:
    X, y, groups = fit_probe.build_dataset(
        cache, labels, "music", "MUSIC_FAKE_PROB", "MUSIC_PRESENT_PROB")
    cv = fit_probe.cross_validate(X, y, groups)
    probe = fit_probe.fit(X, y)
    print("\n[음악 프로브]")
    print(fit_probe.describe(probe, X, y, cv))
    fit_probe.save(MUSIC_HEAD, probe)
    music_probe = script.LinearProbe(MUSIC_HEAD)

# FILE 프로브는 가짜 음악이 없어도 학습된다 (음성 위조만으로도 FILE 라벨이 선다).
# 다만 그 경우 "음악이 진짜일 때"만 배운 프로브다.
Xf, yf, gf = fit_probe.build_dataset(cache, labels, "original", "FILE_FAKE_PROB")
cvf = fit_probe.cross_validate(Xf, yf, gf)
probe_f = fit_probe.fit(Xf, yf)
print("\n[파일 프로브]")
print(fit_probe.describe(probe_f, Xf, yf, cvf))
fit_probe.save(FILE_HEAD, probe_f)
file_probe = script.LinearProbe(FILE_HEAD)

print("\n교차검증 AUC 가 0.5 근처면 임베딩에 신호가 없다는 뜻이다 — 프로브를 쓰지 않는다.")

# ---------- B5 ----------
probe_configs = [(best["name"], best["config"])] + sweep_mod.probe_grid()
results3 = sweep_mod.sweep(script, cache, labels, probe_configs,
                           music_probe=music_probe, file_probe=file_probe,
                           baseline_name=best["name"])

print("\n최고 설정:", results3[0]["name"], results3[0]["config"])
print("\n주의 — 이 검증셋으로 고른 프로브는 같은 생성기에 과적합했을 수 있다.")
print("      학습에 쓰지 않은 생성기로 만든 홀드아웃에서 이득이 유지되는지 반드시 확인한다.")
