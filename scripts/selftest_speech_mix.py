"""Check bundle provenance and mixture math without GPU inference."""
import ast
import hashlib
import io
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np
from scipy.io import wavfile
from run_speech_mix_check import mix, safe_level, unit_rms

ROOT = Path(__file__).resolve().parents[1]


class SpeechMixTests(unittest.TestCase):
    def test_bundle(self):
        bundle = ROOT / 'speech_mix_bundle.zip'
        nb = json.loads((ROOT / 'notebooks/colab_speech_mix.ipynb').read_text(encoding='utf-8'))
        code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        self.assertIn(hashlib.sha256(bundle.read_bytes()).hexdigest(), code)
        self.assertIn('run_speech_mix_check.py', code)
        self.assertIn('speech_mix_results', code)
        with zipfile.ZipFile(bundle) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(len(z.namelist()), len(set(z.namelist())))
            manifest = json.loads(z.read('probe/manifest.json'))
            self.assertEqual(len({r['speaker'] for r in manifest['speech']}), 8)
            self.assertEqual(len(manifest['music']), 16)
            self.assertEqual(sum(r['file_fake'] for r in manifest['music']), 8)
            self.assertEqual({r['split'] for r in manifest['music']}, {'development'})
            self.assertEqual(hashlib.sha256(z.read('frozen_probe.npz')).hexdigest(), manifest['probe_sha256'])
            for row in manifest['speech']:
                blob = z.read('probe/speech/'+row['local_path'])
                self.assertEqual(hashlib.sha256(blob).hexdigest(), row['sha256'])
                self.assertEqual(blob[:4], b'fLaC')
                self.assertGreaterEqual(row['samples'], 128000)
            for row in manifest['music']:
                blob = z.read('probe/test/'+row['ID']+'.wav')
                self.assertEqual(hashlib.sha256(blob).hexdigest(), row['clip_sha256'])
                sr, audio = wavfile.read(io.BytesIO(blob))
                self.assertEqual((sr, audio.shape), (16000, (128000,)))
            for name in ('run_speech_mix_check.py', 'run_file_probe_experiment.py'):
                self.assertEqual(z.read(name), (ROOT/'scripts'/name).read_bytes())

    def test_mix_ratios_and_peak(self):
        time = np.arange(128000)/16000
        voice = np.sin(2*np.pi*200*time)
        music = np.sin(2*np.pi*500*time)
        for db in (-6, 6):
            audio, scale = mix(voice, music, db)
            expected = (unit_rms(voice)*10**(db/20)+unit_rms(music))*scale
            np.testing.assert_allclose(audio, expected, atol=1e-8)
            self.assertLessEqual(np.abs(audio).max(), 0.950001)
        impulse = np.zeros(128000)
        impulse[0] = 1
        self.assertAlmostEqual(float(safe_level(impulse).max()), .95, places=6)

    def test_invalid_sources_stop(self):
        for audio in (np.zeros(128000), np.ones(127999), np.full(128000, np.nan)):
            with self.assertRaises(ValueError):
                unit_rms(audio)


if __name__ == '__main__':
    unittest.main()
