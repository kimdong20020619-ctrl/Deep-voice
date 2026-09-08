#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""검증셋에서 CONFIG 조합을 대량으로 비교한다.

핵심 설계 — 비싼 계산은 한 번, 설정 스윕은 공짜:

  [1회, GPU]  파일마다 PANNs(vp, mp) 와 DF-Arena 세그먼트 점수 3종
              (원본 / 음성 스템 / 음악 스템) 을 캐시에 담는다.
  [무한, CPU] gating · file_head · fusion_mode · segment_agg · 게이트 임계값 조합은
              캐시 위에서 즉시 계산된다.

스윕이 실제 추론과 어긋나지 않도록 **submit/script.py 의 process_one_file 을 그대로 호출**하고
채점기만 캐시 기반으로 바꾼다. 로직을 베껴 쓰지 않으므로 divergence 가 생길 수 없다.

주의: segment_agg 는 캐시된 세그먼트 점수 위에서 다시 계산되므로 스윕 가능하다.
      반면 short_pad · max_segments 는 세그먼트 자체를 바꾸므로 캐시가 무효다.
      그 둘은 precompute 를 다시 돌려야 한다.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np

from .metric import evaluate, format_result, rank_score, score_delta_explained

# 캐시된 오디오를 구분하는 표식. 길이로 구분하므로 서로 달라야 한다.
MARK_ORIGINAL = 1000
MARK_VOICE = 1001
MARK_MUSIC = 1002


def load_script(script_path):
    spec = importlib.util.spec_from_file_location("submit_script", str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _marker(size):
    return np.full(size, 0.1, dtype=np.float32)


class CachedScorer:
    """DF-Arena 대역. 미리 계산해둔 세그먼트 점수를 표식으로 구분해 돌려준다."""

    def __init__(self, script, entry):
        self.script = script
        self.entry = entry

    def score(self, audio, kind=None, want_embeddings=False):
        if audio.size <= 1:                      # 게이팅으로 비워둔 성분
            return (0.0, None) if want_embeddings else 0.0

        if audio.size == MARK_VOICE:
            name = "voice"
        elif audio.size == MARK_MUSIC:
            name = "music"
        else:
            name = "original"

        # 무음 게이트를 스윕하려면 여기서 실제 RMS 와 비교해야 한다.
        # precompute 가 각 스템의 RMS 를 함께 저장한다.
        rms = self.entry.get(f"{name}_rms")
        if rms is not None and rms < self.script.silence_threshold():
            return (0.0, None) if want_embeddings else 0.0

        scores = self.entry[f"{name}_segments"]
        embeddings = self.entry.get(f"{name}_embeddings")
        value = self.script.aggregate_segment_scores(scores, self.script.resolve_agg(kind))
        return (value, embeddings) if want_embeddings else value


def precompute(script, label_rows, test_dir, panns, scorer, htdemucs, device,
               want_embeddings=False, progress_every=25,
               want_conformer_input=False, stems=("original", "voice", "music")):
    """파일마다 캐시 항목을 만든다. GPU 가 필요한 유일한 단계다.

    게이팅 임계값을 스윕하려면 모든 파일의 스템 점수가 필요하므로
    기본값은 게이팅과 무관하게 **항상 분리**한다.

    stems 를 ("original",) 로 좁히면 분리를 건너뛴다. 새 구조(혼합 오디오에서 세 판정을
    직접 내는 방식)의 학습 캐시를 만들 때는 스템이 필요 없고, 그러면 파일당 비용이
    0.164 -> 0.059 s/오디오초로 떨어진다.

    want_conformer_input 은 **원본 스템에만** 적용한다. 세 스템 전부 저장하면 용량이
    3배가 되는데, 학습에 쓰는 건 원본이다.
    """
    import time

    test_dir = Path(test_dir)
    by_stem = {p.stem: p for p in test_dir.iterdir() if p.is_file()}
    need_separation = bool(set(stems) & {"voice", "music"})

    capture = ConformerInputCapture(scorer) if want_conformer_input else None
    cache = []
    started = time.perf_counter()
    for index, row in enumerate(label_rows):
        audio_id = row["ID"]
        path = by_stem.get(audio_id)
        if path is None:
            print(f"[warn] 파일 없음: {audio_id}")
            continue
        try:
            audio = script.load_audio_16k(path)
            voice_present, music_present = script.predict_presence(panns, audio)
            if need_separation:
                voice_audio, music_audio = script.separate_voice_and_music(
                    audio, htdemucs, device)
            else:
                voice_audio = music_audio = None

            entry = {"ID": audio_id, "vp": voice_present, "mp": music_present}
            waves = {"original": audio, "voice": voice_audio, "music": music_audio}
            for name in stems:
                wav = waves[name]
                # 원본에만 Conformer 입력을 모은다 — 학습에 쓰는 건 원본이다.
                stem_capture = capture if name == "original" else None
                segments = _segment_scores(script, scorer, wav, want_embeddings, stem_capture)
                entry[f"{name}_segments"] = segments[0]
                entry[f"{name}_rms"] = script.calculate_rms(wav)   # silence_rms 스윕용
                if want_embeddings:
                    entry[f"{name}_embeddings"] = segments[1]
                if stem_capture is not None:
                    entry["conformer_input"] = stem_capture.take()
            cache.append(entry)
        except Exception as error:
            print(f"[warn] {audio_id} 실패: {type(error).__name__}: {error}")

        if progress_every and (index + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"  {index + 1}/{len(label_rows)}  {elapsed:.0f}s "
                  f"({elapsed / (index + 1):.2f}s/파일)")

    if capture is not None:
        capture.close()
        total = sum(e["conformer_input"].nbytes for e in cache
                    if e.get("conformer_input") is not None)
        print(f"Conformer 입력 캐시 {total / 1024**3:.2f} GB")

    print(f"캐시 {len(cache)}개 완성, {time.perf_counter() - started:.0f}s")
    return cache


class ConformerInputCapture:
    """DF-Arena Conformer 의 입력 (T, 1280) 을 가로채 모아둔다.

    왜 여기에 있는가 — 이건 **학습 캐시를 만들기 위한 것**이지 추론에 필요한 게 아니다.
    제출 zip 의 script.py 에 넣으면 평가 서버가 쓰지도 않을 코드를 들고 가고,
    거기서 나는 오류는 제출 3회 중 1회를 태운다. 학습 쪽에만 둔다.

    왜 Conformer 입력인가 — 그 앞의 XLS-R-1B(얼림)가 연산의 거의 전부다.
    이 지점을 캐시하면 1B 순전파를 세그먼트당 한 번만 하고, 그 뒤 Conformer(~160M)를
    몇십 에폭 학습해도 GPU 시간이 거의 들지 않는다.

    fp16 으로 저장한다. (T=201, D=1280) 이 fp32 면 세그먼트당 1MB, fp16 이면 0.49MB 다.
    """

    def __init__(self, scorer, dtype="float16"):
        self.buffer = []
        self.enabled = False
        self.dtype = dtype
        self.handle = scorer.model.backbone.conformer.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if self.enabled and args:
            self.buffer.append(args[0].detach().float().cpu().numpy().astype(self.dtype))

    def take(self):
        """모아둔 것을 (n_seg, T, D) 로 돌려주고 비운다."""
        if not self.buffer:
            return None
        stacked = np.concatenate(self.buffer, axis=0)
        self.buffer = []
        return stacked

    def close(self):
        self.handle.remove()


def replay_conformer(scorer, conformer_input):
    """캐시한 Conformer 입력을 원본 모델에 다시 넣어 fake 확률을 낸다.

    캐시가 올바른 지점에서 잡혔는지 확인하는 용도다. 이 값이 원래 추론과 다르면
    캐시가 잘못된 것이고 그 위에 올린 학습은 전부 무의미하다. **학습 전에 반드시 통과시킨다.**
    """
    import torch

    tensor = torch.from_numpy(np.asarray(conformer_input)).float().to(scorer.device)
    with torch.inference_mode():
        logits, _ = scorer.model.backbone.conformer(tensor)
    return torch.softmax(logits.float(), dim=-1)[:, scorer.fake_index].cpu().numpy()


def _segment_scores(script, scorer, audio, want_embeddings, capture=None):
    """세그먼트별 원점수를 뽑는다. 집계와 무음 판정은 스윕 때 다시 하므로 여기서는 하지 않는다.

    무음 문턱을 스윕하려면 캐시에 점수가 있어야 하므로, 여기서는 **가장 낮은 문턱**
    (완전 무음만 제외)으로 뽑는다. 실제 문턱 적용은 CachedScorer 가 한다.

    capture 를 주면 Conformer 입력도 함께 모은다 (학습 캐시용).
    """
    if script.calculate_rms(audio) < 1e-9:
        return [], None

    import torch

    segments = script.make_segments(audio)
    scores = []
    embeddings = []
    scorer._capture = bool(want_embeddings) and getattr(scorer, "has_embeddings", False)
    scorer._embeddings = []
    if capture is not None:
        capture.buffer = []
        capture.enabled = True
    try:
        if scorer.batched:
            tensor = torch.from_numpy(segments).to(scorer.device)
            for start in range(0, tensor.shape[0], scorer.batch_size):
                chunk = tensor[start:start + scorer.batch_size]
                scores.extend(scorer._forward_batch(chunk).float().cpu().tolist())
        else:
            for segment in segments:
                tensor = torch.from_numpy(segment).to(scorer.device)
                scores.append(float(scorer._forward_single(tensor)))
    finally:
        if scorer._capture and scorer._embeddings:
            embeddings = np.concatenate(scorer._embeddings, axis=0)
        scorer._capture = False
        scorer._embeddings = []
        if capture is not None:
            capture.enabled = False
    return scores, (embeddings if want_embeddings and len(embeddings) else None)


def predict(script, cache, config, music_probe=None, file_probe=None, multihead=None):
    """캐시 위에서 한 설정의 5개 예측값을 만든다. 실제 process_one_file 을 그대로 쓴다."""
    original_config = copy.deepcopy(script.CONFIG)
    original_load = script.load_audio_16k
    original_presence = script.predict_presence
    original_separate = script.separate_voice_and_music

    script.CONFIG.update(config)
    columns = {name: [] for name in script.PREDICTION_COLUMNS}
    ids = []

    try:
        for entry in cache:
            script.load_audio_16k = lambda _path: _marker(MARK_ORIGINAL)
            script.predict_presence = lambda _p, _a, e=entry: (e["vp"], e["mp"])
            script.separate_voice_and_music = (
                lambda _a, _m, _d: (_marker(MARK_VOICE), _marker(MARK_MUSIC)))

            result = script.process_one_file(
                Path(entry["ID"]), None, CachedScorer(script, entry), None, None,
                music_probe, file_probe, multihead)
            ids.append(entry["ID"])
            for name in script.PREDICTION_COLUMNS:
                columns[name].append(float(result[name]))
    finally:
        script.CONFIG.clear()
        script.CONFIG.update(original_config)
        script.load_audio_16k = original_load
        script.predict_presence = original_presence
        script.separate_voice_and_music = original_separate

    return ids, {name: np.asarray(values) for name, values in columns.items()}


def build_truth(label_rows, ids):
    index = {row["ID"]: row for row in label_rows}
    truth = {name: [] for name in
             ["FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
              "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]}
    for audio_id in ids:
        row = index[audio_id]
        for name in truth:
            truth[name].append(int(row[name]))
    return {name: np.asarray(values) for name, values in truth.items()}


def sweep(script, cache, label_rows, configs, music_probe=None, baseline_name=None,
          file_probe=None, multihead=None):
    """설정 목록을 전부 평가하고 총점 내림차순으로 돌려준다."""
    results = []
    for name, config in configs:
        ids, predictions = predict(script, cache, config, music_probe, file_probe,
                                   multihead)
        truth = build_truth(label_rows, ids)
        result = evaluate(predictions, truth)
        result["name"] = name
        result["config"] = config
        results.append(result)
        print(format_result(result, name))

    results.sort(key=rank_score, reverse=True)

    baseline = None
    if baseline_name:
        baseline = next((r for r in results if r["name"] == baseline_name), None)

    print("\n" + "=" * 100)
    partial = any(np.isnan(r["score"]) for r in results)
    if partial:
        print("주의: 측정 불가한 EER 이 있어 Score* 는 가능한 항목만 재정규화한 값이다")
    print(f"{'순위':<4}{'설정':<26}{'Score*' if partial else 'Score':>9}{'ADS':>9}"
          f"{'file':>8}{'voice':>8}{'music':>8}"
          + ("   기준선 대비 분해" if baseline else ""))
    print("-" * 100)
    for rank, result in enumerate(results, 1):
        line = (f"{rank:<4}{result['name']:<26}{rank_score(result):>9.5f}"
                f"{result['ads_partial'] if partial else result['ads']:>9.5f}"
                f"{result['file_eer']:>8.4f}{result['voice_eer']:>8.4f}{result['music_eer']:>8.4f}")
        if baseline is not None and result["name"] != baseline_name:
            parts = score_delta_explained(baseline, result)
            line += (f"   총 {parts['total']:+.5f}"
                     f" (file {parts['file']:+.4f}, voice {parts['voice']:+.4f},"
                     f" music {parts['music']:+.4f})")
        print(line)
    return results


def default_grid():
    """레버리지 순으로 배치한 기본 스윕 목록.

    실효 가중치: FILE 0.45 · MUSIC 0.27 · VOICE 0.18 · 존재 0.10
    """
    grid = [
        ("baseline(fusion)", {"file_head": "fusion", "fusion_mode": "baseline"}),
        # FILE 0.45 — 가장 무거운 항목
        ("file:direct", {"file_head": "direct"}),
        ("file:direct_max", {"file_head": "direct_max"}),
        ("file:direct_mean", {"file_head": "direct_mean"}),
        ("fuse:noisy_or", {"file_head": "fusion", "fusion_mode": "noisy_or"}),
        ("fuse:gamma0.5", {"file_head": "fusion", "fusion_mode": "gamma", "fusion_gamma": 0.5}),
        ("fuse:gamma0.25", {"file_head": "fusion", "fusion_mode": "gamma", "fusion_gamma": 0.25}),
        ("fuse:gated_max", {"file_head": "fusion", "fusion_mode": "gated_max"}),
        # 공식 경로 — DF-Arena 특징추출기는 앞 64,600 샘플만 본다.
        # 우리의 전 구간 분할+max 가 그보다 나은지 한 번도 측정된 적이 없다.
        # 이게 0-C 하네스 시험의 GPU 가 필요 없는 절반이다.
        ("agg:first(공식경로)", {"segment_agg": "first"}),
        ("agg:first+direct_max", {"segment_agg": "first", "file_head": "direct_max"}),
        # 세그먼트 집계 — 전 항목에 영향
        # max 는 길이 편향이 있다. 4~60초가 섞인 평가셋에서 표본 수가 1~15개로 달라진다.
        # 비율 지정 top-k 는 길이에 따라 k 가 함께 늘어 그 편향을 상쇄한다.
        ("agg:mean", {"segment_agg": "mean"}),
        ("agg:topk2", {"segment_agg": "topk_mean", "segment_topk": 2}),
        ("agg:topk3", {"segment_agg": "topk_mean", "segment_topk": 3}),
        ("agg:top25%", {"segment_agg": "topk_mean", "segment_topk_ratio": 0.25}),
        ("agg:top40%", {"segment_agg": "topk_mean", "segment_topk_ratio": 0.40}),
        ("agg:music_mean", {"segment_agg_music": "mean"}),
        # 무음 게이트 — 낮으면 분리 잔여물(잡음)까지 채점한다
        ("silence:3e-4", {"silence_rms": 3e-4}),
        ("silence:1e-3", {"silence_rms": 1e-3}),
        # 게이팅 임계값 — 속도와 정확도를 함께 바꾼다.
        # 음악 없는 파일의 MUSIC_PRESENT_PROB 실측이 0.060 이라 0.05 이하는 게이트가 죽는다.
        # 0.10 위쪽에서 어디가 최적인지가 실제 질문이다.
        ("gate:off", {"demucs_gating": False}),
        ("gate:m0.05(무효)", {"gate_music": 0.05}),
        ("gate:m0.20", {"gate_music": 0.20}),
        ("gate:m0.30", {"gate_music": 0.30}),
        ("gate:m0.50", {"gate_music": 0.50}),
        ("gate:v0.10", {"gate_voice": 0.10}),
        ("gate:v0.40", {"gate_voice": 0.40}),
    ]
    return grid


def probe_grid():
    """프로브를 학습한 뒤 쓰는 스윕. default_grid 와 이어붙여 쓴다.

    **file_head 재평가가 핵심이다.** 제출 #2 에서 direct 가 fusion 을 이긴 것은
    "음악 가지가 쓰레기일 때"의 결론이었다. 음악 프로브가 서면 융합이 다시 이길 수 있고,
    그러면 MUSIC(0.27) 뿐 아니라 FILE(0.45)까지 회수된다. 그게 프로브를 다는 이유다.

    호출 쪽에서 sweep(..., music_probe=..., file_probe=...) 로 프로브를 넘겨야
    이 설정들이 실제로 동작한다. 프로브가 None 이면 전부 기준선과 같은 값이 나온다.
    """
    return [
        # 음악 프로브 단독 — MUSIC 0.27
        ("probe:music", {"music_head": "probe", "music_head_blend": 1.0}),
        ("probe:music blend0.7", {"music_head": "probe", "music_head_blend": 0.7}),
        ("probe:music blend0.5", {"music_head": "probe", "music_head_blend": 0.5}),
        # 음악이 살아난 뒤의 file_head 재평가 — 여기가 본 게임이다
        ("probe:music+fusion",
         {"music_head": "probe", "file_head": "fusion", "fusion_mode": "baseline"}),
        ("probe:music+gated_max",
         {"music_head": "probe", "file_head": "fusion", "fusion_mode": "gated_max"}),
        ("probe:music+direct_max", {"music_head": "probe", "file_head": "direct_max"}),
        ("probe:music+direct_mean", {"music_head": "probe", "file_head": "direct_mean"}),
        # FILE 프로브 — 0.45 를 직접 겨냥한다. 과적합 위험이 가장 크다
        ("probe:file", {"file_probe": "probe", "file_head": "direct"}),
        ("probe:file blend0.5", {"file_probe": "probe", "file_head": "direct",
                                 "file_probe_blend": 0.5}),
        ("probe:file+direct_max", {"file_probe": "probe", "file_head": "direct_max"}),
        # 둘 다
        ("probe:both", {"music_head": "probe", "file_probe": "probe",
                        "file_head": "direct"}),
        ("probe:both+direct_max", {"music_head": "probe", "file_probe": "probe",
                                   "file_head": "direct_max"}),
    ]


def cross_grid(best_config, extra):
    """최고 설정 위에 추가 변형을 얹은 목록을 만든다 (2단계 스윕용)."""
    out = []
    for name, override in extra:
        config = dict(best_config)
        config.update(override)
        out.append((name, config))
    return out
