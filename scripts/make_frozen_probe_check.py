"""Build a fixed-weight amplitude/pipeline test using development sources only."""
import copy
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
results = ROOT / 'data/experiments/file-probe-20260915/file_probe_results.zip'
with zipfile.ZipFile(results) as z:
    previous = json.loads(z.read('manifest.json'))
    probe = z.read('experimental_file_probe.npz')
    selection = z.read('selection.json')
sources = [r for r in previous['rows'] if r['split'] == 'development']
assert len(sources) == 16
rows, audio_files = [], {}
with zipfile.ZipFile(ROOT / 'file_probe_bundle.zip') as original_bundle:
    for row in sources:
        reference = original_bundle.read('probe/test/' + row['ID'] + '.wav')
        sr, normalized = wavfile.read(io.BytesIO(reference))
        source_blob = (ROOT / 'data/raw/file-probe-20260915/originals' / (row['ID'] + '.wav')).read_bytes()
        assert hashlib.sha256(source_blob).hexdigest() == row['source_sha256']
        raw_sr, raw = wavfile.read(io.BytesIO(source_blob))
        assert sr == raw_sr == 16000 and raw.ndim == 1
        if raw.dtype == np.int16:
            raw = raw.astype(np.float32)/32768
        start = (len(raw)-128000)//2
        raw = raw[start:start+128000].astype(np.float32)
        for condition in ('reference', 'float_reference', 'quiet', 'original_level'):
            identity = row['ID'] + '_' + condition
            if condition == 'reference':
                blob = reference
            else:
                audio = raw if condition == 'original_level' else normalized.astype(np.float32)/32768
                if condition == 'quiet':
                    audio = audio*0.1
                assert audio.shape == (128000,) and np.isfinite(audio).all()
                buffer = io.BytesIO()
                wavfile.write(buffer, 16000, audio)
                blob = buffer.getvalue()
            audio_files[identity] = blob
            rows.append(dict(row, ID=identity, original_ID=row['ID'], condition=condition,
                             clip_sha256=hashlib.sha256(blob).hexdigest()))
manifest = dict(previous, rows=rows, purpose='frozen_development_amplitude_and_pipeline_diagnostic',
                probe_sha256=hashlib.sha256(probe).hexdigest(),
                conditions={'reference': 'exact training preprocessing', 'float_reference': 'same reference samples in float WAV', 'quiet': 'reference x 0.1, float WAV',
                            'original_level': 'source level centered crop, float WAV without clipping'})
out = ROOT / 'frozen_probe_bundle.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    with zipfile.ZipFile(ROOT / 'file_probe_bundle.zip') as base:
        for name in base.namelist():
            if not name.startswith('probe/') and name != 'run_file_probe_experiment.py':
                z.writestr(name, base.read(name))
        for name in ('MUSAN_LICENSE.txt', 'MUSAN_ANNOTATIONS.txt', 'ATTRIBUTION.txt'):
            z.writestr('probe/'+name, base.read('probe/'+name))
    z.writestr('probe/CHANGES.txt', 'Additional diagnostic changes: reference x 0.1 gain and original-level centered 8s float WAV; no fitting.\n')
    z.writestr('frozen_probe.npz', probe)
    z.writestr('selection.json', selection)
    z.writestr('probe/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    for identity, blob in audio_files.items():
        z.writestr('probe/test/'+identity+'.wav', blob)
    for name in ('run_file_probe_experiment.py', 'run_frozen_probe_check.py'):
        z.write(ROOT / 'scripts' / name, name)
nb = json.loads((ROOT / 'notebooks/colab_file_probe.ipynb').read_text(encoding='utf-8'))
nb['cells'][0]['source'] = ['# 고정 판별기의 음량·실제 추론 경로 검사\n',
    '학습을 다시 하지 않습니다. 개발 원본 16개를 네 조건으로 만든 64개 입력입니다.\n',
    '이전 확인셋은 사용하지 않습니다. 독립 성능 검증이나 대회 총점 평가가 아닙니다.\n',
    'reference=기존 정규화, quiet=reference 음량 1/10, original_level=원본 음량입니다.\n',
    'float_reference는 같은 reference 파형을 float WAV로 저장한 대조 조건입니다. quiet와 original_level도 float WAV입니다.\n',
    '파일 판별기를 실제 분리·게이팅 경로에 넣었을 때 단독 계산과 같은지도 검사합니다.\n',
    'frozen_probe_bundle.zip을 올리고 완료 후 frozen_probe_results.zip을 보내주세요.\n']
nb['cells'][2]['source'] = ['## 1. frozen_probe_bundle.zip 업로드\n']
upload = ''.join(nb['cells'][3]['source'])
old_hash = upload.split('hexdigest() == "')[1].split('"')[0]
upload = upload.replace(old_hash, hashlib.sha256(out.read_bytes()).hexdigest()).replace('file_probe_bundle.zip','frozen_probe_bundle.zip')
nb['cells'][3]['source'] = upload.splitlines(True)
nb['cells'][-2]['source'] = ['## 4. 고정 가중치 비교 및 결과 다운로드\n',
    '전체 경로를 반복 실행하므로 이전 특징 추출보다 오래 걸릴 수 있습니다. 오류 시 결과 로그를 보내주세요.\n']
runner = ''.join(nb['cells'][-1]['source']).replace('run_file_probe_experiment.py','run_frozen_probe_check.py').replace('probe_results','frozen_probe_results')
# The preceding replacement also changes file_probe_results; use one clear output name.
runner = runner.replace('file_frozen_probe_results','frozen_probe_results')
nb['cells'][-1]['source'] = runner.splitlines(True)
target = ROOT / 'notebooks/colab_frozen_probe.ipynb'
target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(target, 'bundle bytes', out.stat().st_size, 'rows', len(rows))
