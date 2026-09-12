"""공식 ZIP의 목록과 소량 표본만 읽는다. 전체 데이터·자동 라벨 생성은 하지 않는다."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import struct
import urllib.request
import zipfile

URL = "https://zenodo.org/records/15063698/files/FakeMusicCaps.zip?download=1"
SIZE = 12889873014


def is_audio(entry):
    path = PurePosixPath(entry.filename)
    return (not entry.is_dir() and path.suffix.lower() == ".wav"
            and "__MACOSX" not in path.parts and not path.name.startswith("._"))


def wav_info(blob):
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("Sample has no RIFF/WAVE header")
    offset, fmt, data_size = 12, None, None
    while offset + 8 <= len(blob):
        tag, size = struct.unpack_from("<4sI", blob, offset)
        start = offset + 8
        if start + size > len(blob):
            raise ValueError("Truncated WAV chunk")
        if tag == b"fmt " and size >= 16:
            fmt = struct.unpack_from("<HHIIHH", blob, start)
        if tag == b"data":
            data_size = size
        offset = start + size + size % 2
    if fmt is None or data_size is None or not fmt[2] or not fmt[4]:
        raise ValueError("Missing WAV metadata")
    return {"format_tag": fmt[0], "channels": fmt[1], "sample_rate": fmt[2],
            "bits_per_sample": fmt[5], "seconds": data_size / fmt[4] / fmt[2]}


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, budget=32 * 1024**2):
        self.url, self.size, self.budget = url, size, budget
        self.position = 0
        self.transferred = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("Invalid seek")
        self.position = position
        return position

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b""
        if count > 8 * 1024**2 or self.transferred + count > self.budget:
            raise ValueError("Metadata/sample transfer budget exceeded")
        end = self.position + count - 1
        request = urllib.request.Request(self.url, headers={
            "Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity",
            "User-Agent": "DeepVoice-dataset-audit/1.0"})
        for attempt in range(3):
            if self.transferred + count > self.budget:
                raise ValueError("Metadata/sample transfer budget exceeded")
            with urllib.request.urlopen(request, timeout=30) as response:
                expected = f"bytes {self.position}-{end}/{self.size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Server did not honor byte range; refusing full ZIP download")
                data = response.read(count)
            self.transferred += len(data)
            if len(data) == count:
                break
            if attempt == 2:
                raise ValueError(f"Short range response: expected {count}, got {len(data)}")
        self.position += len(data)
        return data


def inspect(out_dir, samples=0):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Use a new output directory: {out_dir}")
    remote = RemoteZip(URL, SIZE)
    with zipfile.ZipFile(remote) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        audio = [entry for entry in entries if is_audio(entry)]
        summary = {
            "source": "https://zenodo.org/records/15063698",
            "url": URL, "archive_bytes": SIZE, "license": "CC-BY-NC-4.0",
            "archive_md5_published": "db418dc95ab7dc378a55f29d6021fd66",
            "archive_full_checksum_verified": False,
            "files": [{"path": e.filename, "bytes": e.file_size,
                       "compressed_bytes": e.compress_size, "crc32": f"{e.CRC:08x}"}
                      for e in entries],
            "samples": [],
        }
        chosen = []
        # 각 폴더의 첫 표본만 조사한다. 학습·검증 표본 추출 방식이 아니다.
        parents = set()
        for entry in sorted(audio, key=lambda item: item.filename):
            parent = entry.filename.rpartition("/")[0]
            if parent not in parents and len(chosen) < samples:
                chosen.append(entry)
                parents.add(parent)
        downloaded = []
        for index, entry in enumerate(chosen):
            print(f"Inspecting {entry.filename}", flush=True)
            if entry.file_size > 8 * 1024**2:
                raise ValueError(f"Sample too large: {entry.filename}")
            blob = archive.read(entry)
            safe_name = f"sample_{index:02d}.wav"
            downloaded.append((safe_name, blob))
            summary["samples"].append({
                "archive_path": entry.filename, "local_path": safe_name,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "audio_header": wav_info(blob),
                "voice_present": "unreviewed", "music_present": "unreviewed",
                "eligible_for_training": False,
            })
    summary["transferred_bytes"] = remote.transferred
    out_dir.mkdir(parents=True)
    for name, blob in downloaded:
        (out_dir / name).write_bytes(blob)
    (out_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    folders = {}
    for entry in audio:
        parent = entry.filename.rpartition("/")[0]
        folders[parent] = folders.get(parent, 0) + 1
    print(json.dumps({"audio_files": len(audio), "folders": folders,
                      "samples": summary["samples"], "transferred_bytes": remote.transferred,
                      "manifest": str(out_dir / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=0, choices=range(0, 6))
    args = parser.parse_args()
    inspect(args.out_dir, args.samples)
