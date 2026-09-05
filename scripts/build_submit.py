#!/usr/bin/env python3
"""submit/ 를 대회 규격 submit.zip 으로 묶고 제출 전 자가검증한다.

구조가 어긋나면 "설치 오류"가 나고, 그건 일일 제출 횟수를 소모하지 않는다.
하지만 실행 중 오류는 "제출 오류"로 하루 3회 중 1회를 태운다.
그래서 zip 을 만들기 전에 잡을 수 있는 것은 전부 여기서 잡는다.

사용:
    python scripts/build_submit.py                 # 검증 + zip 생성
    python scripts/build_submit.py --check-only    # 검증만
"""

import argparse
import ast
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUBMIT_DIR = REPO / "submit"
OUTPUT_ZIP = REPO / "submit.zip"

MAX_ZIP_BYTES = 10 * 1024 ** 3          # 10 GB
MAX_UNCOMPRESSED_BYTES = 32 * 1024 ** 3  # 32 GB

# zip 루트에 있어야 하는 것. 이 셋 말고 최상위 폴더로 한 번 더 감싸면 설치 오류다.
REQUIRED_TOP_LEVEL = {"model", "script.py", "requirements.txt"}

# 평가 서버 기본 설치 패키지. requirements.txt 에 적으면 버전 충돌로 설치 오류가 난다.
PREINSTALLED = {
    "torch", "torchaudio", "pandas", "numpy", "scipy", "scikit-learn", "sklearn",
    "joblib", "threadpoolctl", "transformers", "accelerate", "huggingface-hub",
    "huggingface_hub", "safetensors", "sentencepiece", "regex", "einops",
    "librosa", "soundfile", "soxr", "demucs", "panns-inference", "panns_inference",
    "torchlibrosa", "julius", "tqdm", "loguru", "pyyaml", "rich", "matplotlib", "diffq",
}

# 평가 서버는 오프라인이다. 추론 경로에 남아 있으면 안 되는 신호들.
NETWORK_HINTS = [
    "hf_hub_download", "snapshot_download", "requests.get", "urllib.request",
    "urlopen", "wget", "curl ", "torch.hub.load",
]

problems = []
warnings = []


def fail(message):
    problems.append(message)
    print(f"  [FAIL] {message}")


def warn(message):
    warnings.append(message)
    print(f"  [WARN] {message}")


def ok(message):
    print(f"  [ OK ] {message}")


def human(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GB"


def check_structure():
    print("\n[1] 제출 구조")
    if not SUBMIT_DIR.is_dir():
        fail(f"submit/ 가 없다: {SUBMIT_DIR}")
        return

    top_level = {p.name for p in SUBMIT_DIR.iterdir() if not p.name.startswith(".")}
    for required in sorted(REQUIRED_TOP_LEVEL):
        if required in top_level:
            ok(f"{required} 있음")
        else:
            fail(f"{required} 없음 — zip 루트에 반드시 있어야 한다")

    extra = top_level - REQUIRED_TOP_LEVEL
    if extra:
        warn(f"규격 외 최상위 항목 {sorted(extra)} — zip 에 포함되지만 규격은 3개다")

    model_dir = SUBMIT_DIR / "model"
    if model_dir.is_dir():
        weights = [p for p in model_dir.rglob("*")
                   if p.is_file() and p.suffix in {".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".th"}]
        if weights:
            ok(f"model/ 가중치 {len(weights)}개, "
               f"{human(sum(p.stat().st_size for p in weights))}")
        else:
            fail("model/ 에 가중치 파일이 없다 — open.zip 의 baseline_submit 내용을 넣어야 한다")
        for expected in ("df_arena_1b", "htdemucs", "panns"):
            if (model_dir / expected).exists():
                ok(f"model/{expected} 있음")
            else:
                fail(f"model/{expected} 없음 — script.py 가 이 경로를 찾는다")


def check_requirements():
    print("\n[2] requirements.txt")
    path = SUBMIT_DIR / "requirements.txt"
    if not path.is_file():
        fail("requirements.txt 가 없다")
        return

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)

    if not entries:
        ok("비어 있다 — 설치 10분 제한과 버전 충돌 위험이 모두 없다")
        return

    ok(f"{len(entries)}개 항목")
    for entry in entries:
        name = entry.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
        if name in PREINSTALLED:
            fail(f"'{entry}' 는 평가 서버 기본 설치 패키지다 — 버전 충돌로 설치 오류가 난다")


def check_script():
    print("\n[3] script.py 정적 검사")
    path = SUBMIT_DIR / "script.py"
    if not path.is_file():
        fail("script.py 가 없다")
        return

    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source)
        ok("구문 검사 통과")
    except SyntaxError as error:
        fail(f"구문 오류 {error.lineno}행: {error.msg}")
        return

    if 'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"' in source:
        ok("출력 경로가 output/submission.csv")
    elif "submission.csv" in source:
        warn("submission.csv 는 있으나 경로 표현을 확인하라 — output/submission.csv 여야 한다")
    else:
        fail("submission.csv 를 쓰는 코드가 없다")

    if 'os.environ["HF_HUB_OFFLINE"] = "1"' in source:
        ok("HF_HUB_OFFLINE 설정됨")
    else:
        warn("HF_HUB_OFFLINE 미설정 — 오프라인 환경에서 HF 접근 시도가 발생할 수 있다")

    for hint in NETWORK_HINTS:
        if hint in source:
            fail(f"네트워크 호출로 보이는 코드: '{hint}' — 평가 서버는 오프라인이다")

    if "try:" in source and "except Exception" in source:
        ok("예외 처리 존재")
    else:
        fail("파일 단위 예외 처리가 없다 — 한 파일 실패가 전체 0점이 된다")

    # 대회 규정: 다른 파일의 정보로 예측을 보정하면 실격이다.
    for banned in ("rankdata", "StandardScaler", "quantile_transform", "zscore"):
        if banned in source:
            fail(f"'{banned}' 발견 — 테스트셋 전체 통계를 쓰는 코드는 규정 위반(실격)이다")


def check_sizes():
    print("\n[4] 용량")
    if not SUBMIT_DIR.is_dir():
        return 0
    files = [p for p in SUBMIT_DIR.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"         파일 {len(files)}개, 압축 전 {human(total)}")
    if total > MAX_UNCOMPRESSED_BYTES:
        fail(f"압축 해제 후 32GB 초과: {human(total)}")
    else:
        ok(f"압축 전 32GB 이내 ({human(total)})")
    return total


def build_zip():
    print("\n[5] zip 생성")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    files = sorted(p for p in SUBMIT_DIR.rglob("*") if p.is_file())
    # 가중치는 이미 압축된 형식이라 재압축 이득이 거의 없다. STORED 가 훨씬 빠르다.
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            archive.write(path, path.relative_to(SUBMIT_DIR).as_posix())

    size = OUTPUT_ZIP.stat().st_size
    print(f"         {OUTPUT_ZIP} — {human(size)}")
    if size > MAX_ZIP_BYTES:
        fail(f"zip 이 10GB 초과: {human(size)}")
    else:
        ok(f"zip 10GB 이내 ({human(size)})")

    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        roots = {name.split("/")[0] for name in archive.namelist()}
    if roots <= REQUIRED_TOP_LEVEL:
        ok(f"zip 루트 = {sorted(roots)}")
    else:
        fail(f"zip 루트가 규격과 다르다: {sorted(roots)} — 최상위 폴더로 감싸면 설치 오류다")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true", help="zip 을 만들지 않고 검증만")
    args = parser.parse_args()

    print(f"대상: {SUBMIT_DIR}")
    check_structure()
    check_requirements()
    check_script()
    check_sizes()

    if not args.check_only and not problems:
        build_zip()
    elif problems:
        print("\n[5] zip 생성 건너뜀 — 먼저 위 실패 항목을 고쳐라")

    print("\n" + "=" * 60)
    print(f"실패 {len(problems)}건, 경고 {len(warnings)}건")
    if problems:
        for item in problems:
            print(f"  - {item}")
        return 1
    print("제출 준비 완료")
    print("남은 검증은 GPU 환경에서만 가능하다 — 더미 3파일 실행과 추론 시간 측정.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
