#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kaggle_valset.ipynb 의 코드 셀을 붙여넣기용 두 덩어리로 합친다.

Kaggle 의 Import Notebook 이 안 될 때를 위한 우회로다.
노트북과 같은 원본(scripts/make_valset_notebook.py)에서 나오므로 내용이 어긋나지 않는다.

A 블록: 환경확인 -> 번들 -> 소스수집 -> 검증셋 -> 모델 -> 캐시   (GPU, 오래 걸림)
B 블록: 설정 스윕 -> 2단계 -> 최종 순위                         (CPU, 몇 초)

A 를 먼저 돌리고, B 는 몇 번이든 다시 돌릴 수 있다.
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO / "notebooks" / "kaggle_valset.ipynb"
OUT_DIR = REPO / "notebooks"

# 노트북 코드 셀 순서 (0-indexed): 0~6 이 A, 7~9 가 B
SPLIT_AT = 8   # 설치 셀이 추가되어 하나 밀렸다


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    print(f"코드 셀 {len(code_cells)}개")

    blocks = {
        "A": ("1단계 — 준비와 채점 (GPU 필요, 30~90분)", code_cells[:SPLIT_AT]),
        "B": ("2단계 — 설정 비교 (몇 초, 몇 번이든 다시 실행 가능)", code_cells[SPLIT_AT:]),
    }

    for key, (title, cells) in blocks.items():
        pieces = [
            f"# ===== {key} 블록 : {title} =====",
            "# Kaggle 새 노트북에 이 전체를 붙여넣고 실행한다.",
            "# Import Notebook 이 안 될 때 쓰는 우회로이며 내용은 노트북과 동일하다.",
            "",
        ]
        for index, cell in enumerate(cells, 1):
            source = "".join(cell["source"]).rstrip()
            pieces.append(f"\n# ---------- {key}{index} ----------")
            pieces.append(source)

        path = OUT_DIR / f"kaggle_paste_{key}.py"
        path.write_text("\n".join(pieces) + "\n", encoding="utf-8")
        lines = sum(1 for _ in path.read_text(encoding="utf-8").splitlines())
        print(f"  {path.name}  ({len(cells)}셀, {lines}줄)")

    print("\n메모장으로 열어 전체 선택(Ctrl+A) -> 복사(Ctrl+C) -> Kaggle 셀에 붙여넣기")


if __name__ == "__main__":
    main()
