"""Download a bounded, provenance-labelled FILE-only diagnostic pilot. No training."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import urllib.request
import zipfile

import numpy as np
from scipy.io import wavfile

from inspect_fakemusiccaps import RemoteZip, URL, SIZE, is_audio

REPO = Path(__file__).resolve().parents[1]
MUSAN_URL = "https://openslr.trmal.net/resources/17/musan.tar.gz"


class LimitedReader:
    def __init__(self, response, budget):
        self.response, self.remaining = response, budget

    def read(self, size=-1):
        if size < 0 or size > self.remaining:
            raise ValueError("MUSAN transfer budget exceeded")
        data = self.response.read(size)
        self.remaining -= len(data)
        return data


def crop(blob):
    sr, audio = wavfile.read(io.BytesIO(blob))
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("Expected mono 16kHz source")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768
    elif audio.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"Unsupported audio dtype: {audio.dtype}")
    if len(audio) < 8 * sr or not np.isfinite(audio).all():
        raise ValueError("Short or nonfinite source")
    start = (len(audio) - 8 * sr) // 2
    clip = audio[start:start + 8 * sr]
    if float(np.sqrt(np.mean(clip**2))) < 1e-6:
        raise ValueError("Silent crop")
    if np.max(np.abs(clip)) > 1.0001:
        raise ValueError("Out-of-range source; do not silently clip")
    pcm = np.rint(np.clip(clip, -1, 32767 / 32768) * 32768).astype(np.int16)
    return pcm, start / sr


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / "originals").mkdir()
    (out / "test").mkdir()
    rows = []

    def save(blob, label, source, group, license_ref):
        pcm, start = crop(blob)
        identity = f"P{len(rows):03d}"
        (out / "originals" / f"{identity}.wav").write_bytes(blob)
        target = out / "test" / f"{identity}.wav"
        wavfile.write(target, 16000, pcm)
        rows.append({"ID": identity, "file_fake": label, "source": source,
                     "source_group": group, "license_reference": license_ref,
                     "source_sha256": hashlib.sha256(blob).hexdigest(),
                     "clip_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                     "crop_start_seconds": start, "duration_seconds": 8,
                     "component_labels_reviewed": False})
        print(identity, label, source, flush=True)

    metadata = REPO / "data/raw/musan-metadata-20260912"
    manifest = json.loads((metadata / "manifest.json").read_text(encoding="utf-8"))
    annotation = next(x for x in manifest["metadata"] if x["path"] == "music/fma/ANNOTATIONS")
    license_item = next(x for x in manifest["metadata"] if x["path"] == "music/fma/LICENSE")
    for item in (annotation, license_item):
        blob = (metadata / item["local_file"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == item["sha256"]
        (out / Path(item["path"]).name).write_bytes(blob)
    artists = {line.split()[0]: line.split()[3] for line in
               (metadata / annotation["local_file"]).read_text(encoding="utf-8").splitlines() if line.strip()}
    licensing = (metadata / license_item["local_file"]).read_text(encoding="utf-8")
    with urllib.request.urlopen(MUSAN_URL, timeout=30) as response:
        bounded = LimitedReader(response, 128 * 1024**2)
        with tarfile.open(fileobj=bounded, mode="r|gz") as archive:
            for entry in archive:
                if not entry.name.startswith("musan/music/fma/") or not entry.name.endswith(".wav"):
                    continue
                track = Path(entry.name).stem
                if track not in artists or track not in licensing:
                    continue
                if not 0 < entry.size < 16 * 1024**2:
                    raise ValueError("Unexpected MUSAN member size")
                save(archive.extractfile(entry).read(), 0, entry.name, artists[track], license_item["url"])
                if len(rows) == 10:
                    break
    if len(rows) != 10:
        raise ValueError("Insufficient real examples")
    remote = RemoteZip(URL, SIZE, budget=32 * 1024**2)
    with zipfile.ZipFile(remote) as archive:
        entries = [e for e in archive.infolist() if is_audio(e)]
        generators = sorted({e.filename.split('/')[0] for e in entries})
        if len(generators) != 5:
            raise ValueError("Unexpected generator inventory")
        used = set()
        for generator in generators:
            candidates = sorted([e for e in entries if e.filename.startswith(generator + '/')], key=lambda e: e.filename)
            random.Random("deepvoice-pilot-v1:" + generator).shuffle(candidates)
            picked = 0
            for entry in candidates:
                group = Path(entry.filename).stem
                if group in used:
                    continue
                if entry.file_size > 2 * 1024**2:
                    raise ValueError("Unexpected fake member size")
                save(archive.read(entry), 1, entry.filename, generator + ':' + group,
                     "https://zenodo.org/records/15063698 (CC BY-NC 4.0)")
                used.add(group)
                picked += 1
                if picked == 2:
                    break
    assert len(rows) == 20
    result = {"purpose": "diagnostic_only_not_training_or_final_holdout", "musan_url": MUSAN_URL,
              "fake_url": URL, "full_archives_checksum_verified": False, "rows": rows}
    (out / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "file_fake"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    prepare(parser.parse_args().out_dir)
