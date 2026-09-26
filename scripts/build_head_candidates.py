# -*- coding: utf-8 -*-
"""학습 노트북 출력(DVHEAD 줄)에서 헤드를 복원하고 제출 후보 zip 을 만든다.

노트북 [9] 셀이 찍은 `DVHEAD <이름> <sha256> <base64>` 줄을 텍스트 파일로 받아
해시를 확인한 뒤 submit/model/ 에 둔다. 후보마다 CONFIG 를 바꿔 build_submit.py 로
빌드하고 script.py 는 항상 원상 복구한다. 기존 zip 은 덮어쓰지 않는다.

사용:
    python scripts/build_head_candidates.py <DVHEAD 텍스트> <후보이름>...
후보 이름은 CANDIDATES 의 키.
"""
import base64
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "submit" / "script.py"
MODEL = ROOT / "submit" / "model"

# 기준은 현재 최고점 제출(topk25). 후보마다 그 위에 변경 하나만 얹는다.
BASE = [('"segment_agg": "max",', '"segment_agg": "topk_mean",'),
        ('"segment_topk_ratio": 0.0,', '"segment_topk_ratio": 0.25,')]
CANDIDATES = {
    "musicprobe": [('"music_head": "none",', '"music_head": "probe",')],
    "musicprobe05": [('"music_head": "none",', '"music_head": "probe",'),
                     ('"music_head_blend": 1.0,', '"music_head_blend": 0.5,')],
    "fileprobe": [('"file_probe": "none",', '"file_probe": "probe",')],
    "fileprobe05": [('"file_probe": "none",', '"file_probe": "probe",'),
                    ('"file_probe_blend": 1.0,', '"file_probe_blend": 0.5,')],
    "direct3": [('"pipeline": "separate",', '"pipeline": "direct",')],
}
HEAD_FILES = {"music_head.npz", "file_head.npz", "head_file.npz", "head_voice.npz", "head_music.npz"}


def restore_heads(text_path):
    restored = []
    for line in Path(text_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("DVHEAD "):
            continue
        _, name, digest, payload = line.split(" ", 3)
        assert name in HEAD_FILES, name
        blob = base64.b64decode(payload.strip())
        assert hashlib.sha256(blob).hexdigest() == digest, f"{name} 해시 불일치 — 복사 중 잘렸다"
        (MODEL / name).write_bytes(blob)
        restored.append(name)
    assert restored, "DVHEAD 줄이 없다"
    return restored


def build(name, original):
    source = original
    for old, new in BASE + CANDIDATES[name]:
        assert source.count(old) == 1, (name, old)
        source = source.replace(old, new)
    SCRIPT.write_text(source, encoding="utf-8", newline="")
    out = ROOT / f"submit_topk25_{name}.zip"
    assert not out.exists(), f"{out.name} 이미 있다 — 덮어쓰지 않는다"
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "scripts/build_submit.py", "--out", str(out)],
                            cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    tail = [l for l in result.stdout.splitlines() if any(k in l for k in ("FAIL", "WARN", "실패", "zip"))]
    print(f"== {out.name} rc={result.returncode}\n" + "\n".join(tail))
    if result.returncode != 0:
        print(result.stdout[-2000:], result.stderr[-1000:])


def main():
    text_path, names = sys.argv[1], sys.argv[2:]
    assert names and all(n in CANDIDATES for n in names), f"후보: {sorted(CANDIDATES)}"
    print("복원:", restore_heads(text_path))
    backup = ROOT / "submit" / ".script.py.bak_candidates"
    shutil.copyfile(SCRIPT, backup)
    original = SCRIPT.read_text(encoding="utf-8")
    try:
        for name in names:
            build(name, original)
    finally:
        shutil.copyfile(backup, SCRIPT)
        backup.unlink()
        print("script.py 복구")


if __name__ == "__main__":
    main()
