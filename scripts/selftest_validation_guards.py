"""후보 선정 오류에 대한 CPU 회귀 검사. 모델 성능 검증은 아니다."""

import copy
import tempfile
import importlib.util
import sys
import types
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from selftest_sweep import make_cache_and_labels, install_stubs, REPO, sweep_mod
from eval.metric import rank_score

# 로컬에 오디오 I/O가 없으면 대체한다. 실제 디코딩·저장은 이 검사 범위가 아니다.
if importlib.util.find_spec("soundfile") is None:
    sys.modules["soundfile"] = types.SimpleNamespace(read=None, write=None)
from synth import build_valset


class ValidationGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_stubs()
        # SciPy는 로드된 torch 모듈의 Tensor 타입도 조회한다.
        if not hasattr(sys.modules["torch"], "Tensor"):
            sys.modules["torch"].Tensor = type("StubTensor", (), {})
        cls.script = sweep_mod.load_script(REPO / "submit" / "script.py")

    def setUp(self):
        self.cache, self.labels = make_cache_and_labels()

    def test_partial_score_cannot_win(self):
        with self.assertRaises(ValueError):
            rank_score({"score": float("nan"), "score_partial": 0.99})

    def test_missing_fake_music_stops_sweep(self):
        for row in self.labels:
            row["MUSIC_FAKE_PROB"] = 0
        with self.assertRaises(ValueError):
            sweep_mod.sweep(self.script, self.cache, self.labels, [("base", {})])

    def test_missing_prediction_stops_sweep(self):
        with patch.object(sweep_mod, "predict") as predict:
            with self.assertRaises(ValueError):
                sweep_mod.sweep(self.script, self.cache[:-1], self.labels, [("base", {})])
            predict.assert_not_called()

    def test_duplicate_and_extra_ids(self):
        ids = [entry["ID"] for entry in self.cache]
        for labels, predictions in [
            (self.labels + [self.labels[0]], ids),
            (self.labels, ids + [ids[0]]),
            (self.labels, ids + ["unknown"]),
        ]:
            with self.subTest(predictions=len(predictions), labels=len(labels)):
                with self.assertRaises(ValueError):
                    sweep_mod.build_truth(labels, predictions)

    def test_fractional_label_rejected(self):
        self.labels[0]["FILE_FAKE_PROB"] = 0.5
        with self.assertRaises(ValueError):
            sweep_mod.build_truth(self.labels, [entry["ID"] for entry in self.cache])

    def test_truth_reorders_without_dropping_samples(self):
        ids = [entry["ID"] for entry in reversed(self.cache)]
        truth = sweep_mod.build_truth(self.labels, ids)
        np.testing.assert_array_equal(truth["FILE_FAKE_PROB"],
                                      [row["FILE_FAKE_PROB"] for row in reversed(self.labels)])

    def test_disabled_probes_do_not_change_baseline(self):
        config = {"music_head": "none", "file_probe": "none", "pipeline": "separate"}
        _, expected = sweep_mod.predict(self.script, self.cache, config)
        _, actual = sweep_mod.predict(self.script, self.cache, config,
                                       music_probe=object(), file_probe=object(), multihead=object())
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_enabled_probe_is_passed_only_when_requested(self):
        probe = object()
        result = {key: 0.5 for key in self.script.PREDICTION_COLUMNS}
        with patch.object(self.script, "process_one_file", return_value=result) as process:
            sweep_mod.predict(self.script, self.cache[:1], {"music_head": "probe"},
                               music_probe=probe)
            self.assertIs(process.call_args.args[5], probe)

    def test_missing_models_fail_and_restore_config(self):
        original = copy.deepcopy(self.script.CONFIG)
        for config in [{"music_head": "probe"},
                       {"file_probe": "probe", "file_head": "direct"},
                       {"pipeline": "direct"}]:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    sweep_mod.predict(self.script, self.cache, config)
                self.assertEqual(self.script.CONFIG, original)

    def test_missing_music_source_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "not-created"
            with self.assertRaises(ValueError):
                build_valset.build({"voice_real": ["a"], "voice_fake": ["b"],
                                    "music_real": ["c"]}, target)
            self.assertFalse(target.exists())

    def test_macos_metadata_source_rejected(self):
        sources = {key: [key + ".wav"] for key in
                   ("voice_real", "voice_fake", "music_real", "music_fake")}
        sources["music_fake"] = ["__MACOSX/musicldm/._track.wav"]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "macOS"):
                build_valset.build(sources, Path(folder) / "unused")

    def test_downsampling_suppresses_aliasing(self):
        sr = 48000
        t = np.arange(sr) / sr
        def convert(hz):
            audio = np.sin(2 * np.pi * hz * t).astype(np.float32)[:, None]
            with patch.object(build_valset.sf, "read", return_value=(audio, sr)):
                return build_valset.load_16k_mono("tone", np.random.default_rng(0), 0.5)
        low, high = convert(1000), convert(12000)
        self.assertEqual(len(low), 16000)
        self.assertGreater(build_valset.rms(low[100:-100]), 0.6)
        # 12kHz 성분이 16kHz 다운샘플 뒤 4kHz로 접히는 회귀를 잡는다.
        if high is not None:
            self.assertLess(build_valset.rms(high[100:-100]), 0.01)

    def test_synthesis_keeps_source_paths(self):
        sources = {key: [key + ".wav"] for key in
                   ("voice_real", "voice_fake", "music_real", "music_fake")}
        tone = np.sin(np.arange(64)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(build_valset, "load_16k_mono", return_value=tone), \
                patch.object(build_valset, "take_span", side_effect=lambda audio, *args: audio), \
                patch.object(build_valset, "mix_sequential", return_value=tone), \
                patch.object(build_valset, "mix_overlap", return_value=tone), \
                patch.object(build_valset, "write_with_format"):
            rows = build_valset.build(sources, folder, count=80, formats=("wav",),
                                      telephone_ratio=0, postprocess_ratio=0)
        for row in rows:
            self.assertEqual(bool(row["voice_source"]), bool(row["VOICE_PRESENT_PROB"]))
            self.assertEqual(bool(row["music_source"]), bool(row["MUSIC_PRESENT_PROB"]))


if __name__ == "__main__":
    unittest.main()
