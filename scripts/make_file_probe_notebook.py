"""Package source-disjoint probe data and a Colab runner, separate from submissions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from run_file_probe_experiment import validate_rows
from prepare_probe_dataset import normalized_crop
from scipy.io import wavfile
import io

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data/raw/file-probe-20260915'
manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
# The original MUSAN licence lists NoDerivatives for these works. Exclude the
# whole artist from cropped training data rather than guessing adaptation rights.
excluded = [r for r in manifest['rows'] if r['group'] == 'Morgan_Sadler' and not r['file_fake']]
manifest['rows'] = [r for r in manifest['rows'] if r not in excluded]
manifest['license_excluded_ids'] = [r['ID'] for r in excluded]
manifest['preprocessing'] = 'center 8s, peak normalize every clip to 0.95, mono 16kHz PCM16'
# Rebuild all clips with one normalization, including files collected before the
# source float WAV with amplitude over 1.0 exposed the encoding assumption.
audio_blobs = {}
for row in manifest['rows']:
    original = (DATA / 'originals' / (row['ID'] + '.wav')).read_bytes()
    assert hashlib.sha256(original).hexdigest() == row['source_sha256']
    pcm, start = normalized_crop(original)
    buffer = io.BytesIO()
    wavfile.write(buffer, 16000, pcm)
    audio_blobs[row['ID']] = buffer.getvalue()
    row['clip_sha256'] = hashlib.sha256(buffer.getvalue()).hexdigest()
    row['crop_start_seconds'] = start
validate_rows(manifest['rows'])
validate_rows(manifest['rows'])
out = REPO / 'file_probe_bundle.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    with zipfile.ZipFile(REPO / 'verify_bundle.zip') as baseline:
        for name in baseline.namelist():
            if not name.startswith('data/'):
                z.writestr(name, baseline.read(name))
    z.write(REPO / 'scripts/run_file_probe_experiment.py', 'run_file_probe_experiment.py')
    z.writestr('probe/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    for name in ('MUSAN_LICENSE.txt', 'MUSAN_ANNOTATIONS.txt'):
        z.write(DATA / name, 'probe/' + name)
    for row in manifest['rows']:
        z.writestr('probe/test/' + row['ID'] + '.wav', audio_blobs[row['ID']])
    z.writestr('probe/ATTRIBUTION.txt', 'MUSAN: Snyder, Chen, Povey (2015), https://www.openslr.org/17/ ; individual artists, titles and terms: MUSAN_LICENSE.txt and MUSAN_ANNOTATIONS.txt.\nFakeMusicCaps: Politecnico di Milano ISPL, https://zenodo.org/records/15063698 ; CC BY-NC 4.0, https://creativecommons.org/licenses/by-nc/4.0/ .\nChanges: 8-second centered crops, peak normalized to 0.95, mono 16kHz PCM16. Noncommercial internal experiment. NoDerivatives source excluded.\n')
bundle_hash = hashlib.sha256(out.read_bytes()).hexdigest()
old = json.loads((REPO / 'notebooks/colab_music_pilot.ipynb').read_text(encoding='utf-8'))
notebook = dict(old, cells=old['cells'][:8])
notebook['cells'][0]['source'] = ['# 음악 파일 판별 학습 실험\n',
    '기존 모델 특징에 작은 분류기를 학습합니다. 전체 모델 미세조정은 하지 않습니다.\n',
    '학습/개발/확인은 연주자·원본·생성 설명·생성기를 분리했습니다. 이전 파일럿 그룹은 제외했습니다.\n',
    '표본이 작은 가능성 확인 실험입니다. FILE EER/AUC만 계산하며 대회 총점으로 해석하지 않습니다.\n',
    '최종 확인 결과를 보고 재튜닝하지 않습니다. 좋은 결과가 나와도 제출 전 별도 검증이 필요합니다.\n',
    'file_probe_bundle.zip을 올리고 위에서부터 실행하세요. 결과 file_probe_results.zip을 보내주세요.\n']
notebook['cells'][2]['source'] = ['## 1. file_probe_bundle.zip 업로드\n']
upload = ''.join(notebook['cells'][3]['source'])
old_hash = upload.split('hexdigest() == "')[1].split('"')[0]
upload = upload.replace(old_hash, bundle_hash).replace('music_pilot_bundle.zip', 'file_probe_bundle.zip').replace('deepvoice_pilot_', 'deepvoice_probe_').replace('pilot/', 'probe/').replace('실험 음원 20개 무결성 확인 완료', '학습 실험 음원 무결성 확인 완료')
notebook['cells'][3]['source'] = upload.splitlines(True)
notebook['cells'].append({'cell_type': 'markdown', 'metadata': {}, 'source': [
    '## 4. 특징 추출 → 학습 → 개발셋 선택 → 확인셋 평가\n',
    '표준화와 학습에는 train만 사용합니다. C=0.001/0.01/0.1 중 개발 EER, AUC, 작은 C 순으로 선택합니다.\n',
    '설정을 선택한 뒤 확인셋을 한 번 평가합니다. 기존 모델과 같은 파일에서 비교합니다.\n',
    '실험 가중치는 자동으로 제출 코드에 적용하지 않습니다. 모델은 한 번 로드하고 파일별 특징을 저장합니다.\n']})
runner = '''import subprocess, sys, shutil, pathlib
worker_results = WORK / "probe_results"
worker_results.mkdir(exist_ok=True)
shutil.copy2(RESULTS / "environment.txt", worker_results / "environment.txt")
log_path = worker_results / "run.log"
try:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, "-u", "run_file_probe_experiment.py", "--work", str(WORK)], cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
    assert returncode == 0, "실험이 중단됐습니다. 결과 ZIP 또는 출력 노트북을 보내주세요."
finally:
    if pathlib.Path("/content/install.log").exists():
        shutil.copy2("/content/install.log", worker_results / "install.log")
    archive = shutil.make_archive("/content/file_probe_results", "zip", worker_results)
    files.download(archive)
'''
notebook['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': runner.splitlines(True)})
target = REPO / 'notebooks/colab_file_probe.ipynb'
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('READY', target, 'bundle_bytes', out.stat().st_size, 'sha256', bundle_hash)
print(Counter((r['split'], r['file_fake']) for r in manifest['rows']))
