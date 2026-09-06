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
print("최종 순위")
for rank, r in enumerate(results2, 1):
    print(f"{rank:>2}. {r['name']:<28} Score {r['score']:.5f}  "
          f"(file {r['file_eer']:.4f} / voice {r['voice_eer']:.4f} / music {r['music_eer']:.4f})")
print("\n최고 설정 CONFIG 덮어쓰기:")
print(results2[0]["config"])
