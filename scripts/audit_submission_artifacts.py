"""Inventory local submission ZIPs without loading models or rewriting archives."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate archive members: {path.name}")
        source = archive.read("script.py")
        tree = ast.parse(source.decode("utf-8-sig"))
        configs = [ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets)]
        if len(configs) != 1:
            raise ValueError(f"Expected one literal CONFIG: {path.name}")
        models = [{"name": entry.filename, "bytes": entry.file_size, "crc32": f"{entry.CRC:08x}"}
                  for entry in entries if entry.filename.startswith("model/") and not entry.is_dir()]
        return {"archive": path.name, "archive_bytes": path.stat().st_size,
                "uncompressed_bytes": sum(entry.file_size for entry in entries),
                "script_sha256": hashlib.sha256(source).hexdigest(), "config": configs[0],
                "models_directory_metadata": models,
                "full_archive_hash_verified": False, "model_payload_hash_verified": False,
                "leaderboard_score": None, "score_mapping_status": "requires_submission_receipt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parents[1]
    paths = sorted(repo.glob("submit*.zip"))
    if not paths:
        raise FileNotFoundError("No submission ZIPs")
    records = [inspect(path) for path in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"artifacts": records,
                   "warning": "ZIP directory CRC/size equality is not proof of identical model bytes or uploaded artifact identity."},
                  handle, ensure_ascii=False, indent=2)
    for record in records:
        print(record["archive"], record["script_sha256"],
              {key: record["config"].get(key) for key in ("file_head", "music_head", "pipeline", "segment_agg")})


if __name__ == "__main__":
    main()
