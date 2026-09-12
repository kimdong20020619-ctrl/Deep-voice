"""Exercise generated notebook evaluation with fake predictions, never fake GPU evidence."""

import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from scipy.io import wavfile

REPO = Path(__file__).resolve().parents[1]


class PilotTests(unittest.TestCase):
    def test_bundle(self):
        with zipfile.ZipFile(REPO / "music_pilot_bundle.zip") as archive:
            self.assertIsNone(archive.testzip())
            manifest = json.loads(archive.read("pilot/manifest.json"))
            self.assertEqual(len(manifest["rows"]), 20)
            self.assertEqual(sum(r["file_fake"] for r in manifest["rows"]), 10)
            for row in manifest["rows"]:
                blob = archive.read("pilot/test/" + row["ID"] + ".wav")
                self.assertEqual(hashlib.sha256(blob).hexdigest(), row["clip_sha256"])
                sr, audio = wavfile.read(io.BytesIO(blob))
                self.assertEqual((sr, audio.shape, audio.dtype), (16000, (128000,), np.dtype('int16')))

    def run_evaluation(self, invalid=False):
        notebook = json.loads((REPO / "notebooks/colab_music_pilot.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(''.join(cell["source"]))
        code = ''.join(notebook["cells"][-1]["source"])
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            results = work / "results"
            results.mkdir()
            (work / "output").mkdir()
            source = (REPO / "submit/script.py").read_text(encoding="utf-8")
            (work / "script.py").write_text(source, encoding="utf-8")
            labels = pd.DataFrame({"ID": ["a", "b"], "file_fake": [0, 1]})
            columns = ["ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]
            seen = []

            def fake_run(*args, **kwargs):
                tree = ast.parse((work / "script.py").read_text(encoding="utf-8"))
                cfg = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
                seen.append((cfg["file_head"], cfg["segment_agg"]))
                pred = pd.DataFrame({"ID": ["b", "wrong" if invalid else "a"]})
                for column in columns[1:]:
                    pred[column] = [0.9, 0.1]
                pred.to_csv(work / "output/submission.csv", index=False)
                return subprocess.CompletedProcess(args, 0, "실패 0건", "")

            namespace = dict(WORK=work, RESULTS=results, Path=Path, labels=labels, columns=columns,
                             pd=pd, np=np, os=os, sys=sys, json=json, hashlib=hashlib, shutil=shutil,
                             files=SimpleNamespace(download=lambda path: None))
            with patch("subprocess.run", side_effect=fake_run), patch("shutil.make_archive", return_value="mock-results.zip"), contextlib.redirect_stdout(io.StringIO()):
                if invalid:
                    with self.assertRaises(AssertionError):
                        exec(code, namespace)
                    self.assertTrue((results / "ERROR.txt").exists())
                else:
                    exec(code, namespace)
                    scores = pd.read_csv(results / "summary.csv")
                    self.assertEqual(seen, [("direct", "max"), ("direct", "mean"), ("fusion", "max")])
                    self.assertTrue((scores.file_eer == 0).all())
                    self.assertTrue((scores.file_auc == 1).all())
            self.assertEqual((work / "script.py").read_text(encoding="utf-8"), source)

    def test_evaluation_alignment_and_config(self):
        self.run_evaluation()

    def test_bad_ids_rejected_and_code_restored(self):
        self.run_evaluation(invalid=True)


if __name__ == "__main__":
    unittest.main()
