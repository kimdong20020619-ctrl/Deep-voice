"""검토 전 FakeMusicCaps 원본 목록을 분할한다. 오디오 확보·라벨 검토 완료를 뜻하지 않는다."""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


GENERATOR_SPLITS = {
    "MusicGen_medium": "train", "musicldm": "train", "mustango": "train",
    "audioldm2": "development", "stable_audio_open": "holdout",
}


def caption_split(caption_id):
    value = int(hashlib.sha256(("deepvoice-v1:" + caption_id).encode()).hexdigest(), 16) % 100
    return "train" if value < 70 else "development" if value < 85 else "holdout"


def plan(manifest, output):
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    rows, seen = [], set()
    groups = {name: set() for name in GENERATOR_SPLITS.values()}
    generators = {name: set() for name in GENERATOR_SPLITS.values()}
    for entry in metadata["files"]:
        path = PurePosixPath(entry["path"])
        if path.suffix.lower() != ".wav" or "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        if len(path.parts) != 2 or path.parts[0] not in GENERATOR_SPLITS:
            raise ValueError(f"Unknown archive layout: {path}")
        if str(path) in seen:
            raise ValueError(f"Duplicate path: {path}")
        seen.add(str(path))
        generator, group = path.parts[0], path.stem
        split = caption_split(group)
        if split != GENERATOR_SPLITS[generator]:
            continue
        groups[split].add(group)
        generators[split].add(generator)
        rows.append({"dataset": "FakeMusicCaps", "revision": "zenodo.15063698",
                     "archive_path": str(path), "caption_group": group,
                     "generator": generator, "planned_split": split,
                     "review_status": "unreviewed", "eligible_for_training": "false",
                     "local_audio_path": "", "audio_sha256": "",
                     "voice_present": "", "music_present": ""})
    for left, right in [("train", "development"), ("train", "holdout"), ("development", "holdout")]:
        if groups[left] & groups[right] or generators[left] & generators[right]:
            raise ValueError("Split leakage")
    if not all(groups.values()):
        raise ValueError("Empty split")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({split: {"candidate_files": sum(r["planned_split"] == split for r in rows),
                             "caption_groups": len(groups[split]),
                             "generators": sorted(generators[split])}
                      for split in groups}, ensure_ascii=False, indent=2))
    print("No audio downloaded or labels approved; plan only:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan(args.manifest, args.output)
