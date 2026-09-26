# -*- coding: utf-8 -*-
"""notebooks/train_heads_run.py 를 백그라운드로 띄우는 Colab 실행용 노트북을 만든다.

09-25 첫 실행은 셀 출력(진행률 위젯) 때문에 브라우저 탭이 멈추고 런타임이 회수돼 결과를 잃었다.
그래서 학습은 nohup 프로세스로 돌리고, 셀은 몇 분마다 로그 꼬리만 덮어써 런타임을 유지한다.

사용: python scripts/make_colab_launcher.py  (먼저 make_kaggle_train_notebook.py 실행)
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "notebooks" / "train_heads_run.py"
OUTPUT = ROOT / "notebooks" / "colab_launch_heads.ipynb"

LAUNCH = '''import base64, os, pathlib, subprocess
src = base64.b64decode("{payload}").decode("utf-8")
path = pathlib.Path("/content/train_heads_run.py")
path.write_text(src, encoding="utf-8")
env = dict(os.environ, HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1")
# MLAAD(게이트) 토큰 — Colab 왼쪽 열쇠 아이콘 Secrets 에 HF_TOKEN 이 있으면 전체 모드, 없으면 음악 전용 모드.
# userdata 는 커널에서만 읽히므로 여기서 읽어 환경변수로 넘긴다.
try:
    from google.colab import userdata
    env["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN 있음 — 전체 모드")
except Exception as error:
    print("HF_TOKEN 없음 — 음악 전용 모드:", type(error).__name__)
log = open("/content/train.log", "ab")
proc = subprocess.Popen(["python", "-u", str(path)], stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True, cwd="/content")
pathlib.Path("/content/train.pid").write_text(str(proc.pid))
print("started pid", proc.pid)
'''

MONITOR = '''# 학습이 끝날 때까지 런타임을 붙잡아 둔다. 출력은 매번 지워 탭이 무거워지지 않게 한다.
import os, time, pathlib
from IPython.display import clear_output
pid = int(pathlib.Path("/content/train.pid").read_text())
def alive(p):
    try:
        os.kill(p, 0)
        return True
    except OSError:
        return False
while alive(pid):
    lines = pathlib.Path("/content/train.log").read_text(errors="ignore").splitlines()
    clear_output(wait=True)
    print(time.strftime("%H:%M:%S"), "running pid", pid)
    print("\\n".join(l[:200] for l in lines[-12:] if not l.startswith("DVHEAD")))
    time.sleep(180)
clear_output(wait=True)
print("finished")
'''

RESULT = '''# 끝난 뒤 실행 — 헤드(base64)와 지표를 찍는다. 이 출력만 옮기면 된다.
import pathlib
for line in pathlib.Path("/content/train.log").read_text(errors="ignore").splitlines():
    if line.startswith(("DVHEAD", "DVMETRICS")) or "== " in line or "[error]" in line or "Traceback" in line:
        print(line)
'''


def main():
    payload = base64.b64encode(RUNNER.read_bytes()).decode("ascii")
    cells = [LAUNCH.format(payload=payload), MONITOR, RESULT]
    for index, cell in enumerate(cells):
        compile(cell, f"cell{index}", "exec")
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c}
                  for c in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU",
                     "colab": {"gpuType": "T4"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    print(f"{OUTPUT} — {OUTPUT.stat().st_size / 1024:.0f} KB, 셀 3개 문법 검사 통과")


if __name__ == "__main__":
    main()
