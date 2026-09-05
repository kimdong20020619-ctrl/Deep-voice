#!/usr/bin/env python3
"""Colab 검증용 verify_bundle.zip 을 만든다.

가중치 4.66GB 를 업로드하는 대신 코드·설정·더미 데이터만 담는다(약 250KB).
큰 가중치는 notebooks/colab_verify.ipynb 가 원본 소스에서 직접 받고
대회 배포본의 SHA256SUMS.txt 로 대조한다.
"""

import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMIT = REPO / "submit"
RAW_DATA = REPO / "data" / "raw" / "data"
OUT = REPO / "verify_bundle.zip"

# Colab 에서 원본 소스로 받을 것들. 번들에 넣지 않는다.
BIG_SUFFIXES = {".bin", ".pth", ".th", ".safetensors", ".ckpt"}


def main():
    assert SUBMIT.is_dir(), f"submit/ 가 없다: {SUBMIT}"
    assert (SUBMIT / "script.py").is_file(), "submit/script.py 가 없다"
    assert RAW_DATA.is_dir(), (
        f"배포 데이터가 없다: {RAW_DATA}\n"
        "open.zip 의 data/ 를 data/raw/data/ 로 복사했는지 확인하라"
    )

    if OUT.exists():
        OUT.unlink()

    picked = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SUBMIT.rglob("*")):
            if not path.is_file() or path.suffix in BIG_SUFFIXES:
                continue
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(SUBMIT).as_posix()
            archive.write(path, rel)
            picked.append((rel, path.stat().st_size))

        # 배포 더미 3개와 제출 양식도 함께 넣는다
        for path in sorted(RAW_DATA.rglob("*")):
            if path.is_file():
                rel = "data/" + path.relative_to(RAW_DATA).as_posix()
                archive.write(path, rel)
                picked.append((rel, path.stat().st_size))

        # 검증셋 노트북이 쓰는 라이브러리 코드. 같은 번들로 Colab/Kaggle 둘 다 커버한다.
        for path in sorted((REPO / "src").rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = "src/" + path.relative_to(REPO / "src").as_posix()
            archive.write(path, rel)
            picked.append((rel, path.stat().st_size))

    print(f"{OUT.name}  {OUT.stat().st_size / 1024:.0f} KB, {len(picked)} 파일")
    for rel, size in picked:
        print(f"  {size / 1024:8.1f} KB  {rel}")

    names = {rel for rel, _ in picked}
    for required in ("script.py", "requirements.txt",
                     "model/panns/component_labels.json",
                     "model/panns/class_labels_indices.csv",
                     "model/SHA256SUMS.txt",
                     "data/sample_submission.csv",
                     "src/eval/metric.py", "src/eval/sweep.py",
                     "src/synth/build_valset.py"):
        assert required in names, f"번들에 {required} 가 빠졌다"
    print("\n필수 파일 확인 완료")


if __name__ == "__main__":
    main()
