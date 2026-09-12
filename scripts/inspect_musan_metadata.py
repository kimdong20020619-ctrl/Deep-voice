"""Read bounded public mirror metadata; never infer audio availability from folders."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request


def fetch(url, limit=2 * 1024**2):
    with urllib.request.urlopen(url, timeout=30) as response:
        blob = response.read(limit + 1)
    if len(blob) > limit:
        raise ValueError(f"Metadata exceeds limit: {url}")
    return blob


def inspect(out_dir):
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(out)
    repo = "FluidInference/musan"
    info_url = f"https://huggingface.co/api/datasets/{repo}"
    info = json.loads(fetch(info_url))
    revision = info["sha"]
    paths = [entry["rfilename"] for entry in info["siblings"]]
    selected = [p for p in paths if p.startswith("music/")
                and PurePosixPath(p).name in {"ANNOTATIONS", "LICENSE", "README"}]
    if len(selected) > 20 or not selected:
        raise ValueError("Unexpected music metadata inventory")
    records, contents = [], []
    for index, path in enumerate(sorted(selected)):
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"
        blob = fetch(url, 256 * 1024)
        blob.decode("utf-8")
        local = f"{index:02d}_{PurePosixPath(path).name}.txt"
        records.append({"path": path, "url": url, "local_file": local,
                        "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
        contents.append((local, blob))
    audio = [p for p in paths if p.startswith("music/")
             and PurePosixPath(p).suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}]
    manifest = {"repository": repo, "revision": revision,
                "metadata_api": info_url, "all_paths": paths,
                "music_audio_paths": audio, "metadata": records,
                "official_archive_bytes_verified": False,
                "eligible_for_training": False}
    out.mkdir(parents=True)
    for name, content in contents:
        (out / name).write_bytes(content)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"revision": revision, "music_audio_count": len(audio),
                      "metadata_count": len(records), "output": str(out)}, ensure_ascii=False))


def analyze(out_dir):
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    records = {item["path"]: item for item in manifest["metadata"]}
    for item in records.values():
        blob = (out / item["local_file"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"Metadata hash mismatch: {item['local_file']}")
    rows, seen = [], set()
    for path, item in sorted(records.items()):
        if not path.endswith("/ANNOTATIONS"):
            continue
        source = PurePosixPath(path).parent.as_posix()
        license_item = records[source + "/LICENSE"]
        license_text = (out / license_item["local_file"]).read_text(encoding="utf-8")
        for line in (out / item["local_file"]).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) not in (4, 5) or fields[2] not in {"Y", "N"}:
                raise ValueError(f"Invalid annotation: {line}")
            track, genres, vocals, artist = fields[:4]
            if track in seen:
                raise ValueError(f"Duplicate track: {track}")
            seen.add(track)
            rows.append({"track_id": track, "source": source, "genres": genres,
                         "annotated_vocals": vocals, "artist_raw": artist,
                         "composer_raw": fields[4] if len(fields) == 5 else "",
                         "artist_group_review": "pending_alias_and_recording_review",
                         "annotation_url": item["url"], "license_url": license_item["url"],
                         "id_mentioned_in_license": str(track in license_text).lower(),
                         "audio_available_in_mirror": str(any(PurePosixPath(p).stem == track
                                                              for p in manifest["music_audio_paths"])).lower(),
                         "eligible_for_training": "false", "planned_split": "",
                         "review_status": "metadata_only"})
    if not rows:
        raise ValueError("No annotations")
    with (out / "music_inventory.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"annotation_rows": len(rows),
                      "annotated_no_vocals": sum(row["annotated_vocals"] == "N" for row in rows),
                      "missing_id_in_license": sum(row["id_mentioned_in_license"] == "false" for row in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if not args.analyze_existing:
        inspect(args.out_dir)
    analyze(args.out_dir)
