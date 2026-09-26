# -*- coding: utf-8 -*-
"""scripts/kaggle_train_heads_src.py 를 Kaggle 노트북으로 만든다.

제출 코드(submit/script.py)와 DF-Arena 소형 코드 파일을 base64 로 넣는다 — Kaggle 에서
추론과 똑같은 전처리·임베딩을 쓰게 하려는 것이다. 대용량 가중치는 노트북이 HF 에서 받아
해시로 대조한다.

사용: python scripts/make_kaggle_train_notebook.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "scripts" / "kaggle_train_heads_src.py"
OUTPUT = ROOT / "notebooks" / "kaggle_train_heads.ipynb"
RUNNER = ROOT / "notebooks" / "train_heads_run.py"
DF_DIR = ROOT / "submit" / "model" / "df_arena_1b"
SKIP = {"pytorch_model.bin", ".gitattributes", ".gitignore"}


def embedded_files():
    files = {"script.py": (ROOT / "submit" / "script.py").read_bytes(),
             "model/htdemucs/htdemucs.yaml": (ROOT / "submit" / "model" / "htdemucs" / "htdemucs.yaml").read_bytes()}
    for path in sorted(DF_DIR.rglob("*")):
        if path.is_file() and path.name not in SKIP:
            files[f"model/df_arena_1b/{path.relative_to(DF_DIR).as_posix()}"] = path.read_bytes()
    return {k: base64.b64encode(v).decode("ascii") for k, v in files.items()}


def main():
    text = SOURCE.read_text(encoding="utf-8")
    payload = base64.b64encode(json.dumps(embedded_files()).encode("utf-8")).decode("ascii")
    assert text.count("__EMBEDDED_FILES__") == 1
    text = text.replace("__EMBEDDED_FILES__", payload)

    cells, current = [], []
    for line in text.splitlines(keepends=True):
        if line.startswith("# %%") and current:
            cells.append(current)
            current = []
        current.append(line)
    cells.append(current)
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": "".join(cell)} for cell in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    # 같은 내용의 단일 스크립트 — 터미널에서 nohup 으로 돌리면 브라우저 탭 상태와 무관하게 끝까지 간다
    RUNNER.write_text(text, encoding="utf-8")
    # 셀마다 문법 검사 — 업로드 후 첫 셀에서 죽으면 GPU 시간만 버린다
    for index, cell in enumerate(cells):
        compile("".join(cell), f"cell{index}", "exec")
    print(f"{OUTPUT} — 셀 {len(cells)}개, {OUTPUT.stat().st_size / 1024:.0f} KB, 문법 검사 통과")


if __name__ == "__main__":
    main()
