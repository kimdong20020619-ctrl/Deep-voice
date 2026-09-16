"""Package existing development inputs for frozen separation/amplitude comparisons."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

from run_separation_check import validate_plan

ROOT = Path(__file__).resolve().parents[1]


def main():
    result_path = ROOT/'data/experiments/mixed-probe-20260916/mixed_probe_results.zip'
    assert hashlib.sha256(result_path.read_bytes()).hexdigest() == '4730fc617099dc423eb17bc9b230474d100490951127a54236f435214addebf1'
    source_path = ROOT/'mixed_probe_bundle.zip'
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == 'fad4f51c51480b7ab4aac30ab22b94441729adeb3ea2ea224a33a7079ee76fef'
    with zipfile.ZipFile(source_path) as base, zipfile.ZipFile(result_path) as result:
        original = json.loads(base.read('probe/manifest.json'))
        sources = [s for s in original['sources'] if s['split']=='development']
        indexed = {s['ID']:s for s in sources}
        rows = []
        for row in original['rows']:
            if row['split'] != 'development':
                continue
            selected = [indexed[s] for s in row['source_ids']]
            eligible = any(s['kind']=='music' for s in selected) and not any(s['kind']=='speech' and s['file_fake'] for s in selected)
            rows.append(dict(row,music_comparison_eligible=eligible))
        assert len(rows)==80 and sum(r['music_comparison_eligible'] for r in rows)==48
        reference = {r['ID']:{k:float(r[k]) for k in ('baseline','previous_music_probe')}
                     for r in csv.DictReader(io.StringIO(result.read('predictions.csv').decode()))}
        frozen = base.read('previous_music_probe.npz')
        assert hashlib.sha256(frozen).hexdigest() == '08bfc83eab9f4dd23a4f3c9c884f0f1c9e25230634ff4245da8d3a289539601f'
        plan = dict(rows=rows,sources=sources,reference_predictions=reference,
                    probe_sha256=hashlib.sha256(frozen).hexdigest(),
                    purpose='Frozen development separation diagnostic; music-trained FILE probe, not a validated component head',
                    views=['original','music_raw','music_level'],
                    forced_separation=True, fit=False, independent_validation=False,
                    limitations=original['limitations'])
        validate_plan(plan)
        output = ROOT/'separation_bundle.zip'
        with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
            for name in base.namelist():
                if name.startswith(('model/','src/')) or name in ('script.py','requirements.txt'):
                    z.writestr(name,base.read(name))
                elif name.startswith('probe/') and name.endswith('.txt'):
                    z.writestr(name,base.read(name))
            z.writestr('probe/SEPARATION_NOTE.txt',
                'Additional changes: forced HTDemucs separation and optional per-file RMS 0.05 scaling, peak guard 0.95. '
                'Derived stems retain each source-specific license; WaveFake-derived stems remain CC BY-SA 4.0. '
                'This diagnostic uses development inputs only; it is not a submission model.\n')
            for row in rows:
                name = 'probe/test/'+row['ID']+'.wav'
                blob = base.read(name)
                assert hashlib.sha256(blob).hexdigest()==row['clip_sha256']
                z.writestr(name,blob)
            z.writestr('probe/manifest.json',json.dumps(plan,ensure_ascii=False,indent=2))
            z.writestr('fixed_music_probe.npz',frozen)
            for name in ('run_separation_check.py','run_file_probe_experiment.py'):
                z.write(ROOT/'scripts'/name,name)
    notebook = json.loads((ROOT/'notebooks/colab_mixed_probe.ipynb').read_text(encoding='utf-8'))
    notebook['cells'][0]['source'] = ['# 음악 분리 전후 비교 — 고정 가중치\n',
        'T4 GPU를 선택하고 모두 실행한 뒤 separation_bundle.zip을 업로드하세요.\n',
        '기존 개발 입력 80개에서 원본·분리 음악·음량을 맞춘 분리 음악을 비교합니다. 재학습하지 않습니다.\n',
        '실제 음성+음악 및 음악 단독 48개만 분리 음악 점수의 파일 오류율 비교에 사용합니다. 나머지 32개는 대조 입력으로 기록합니다.\n',
        'Music EER나 전체 대회 점수가 아닙니다. 결과 separation_results.zip을 보내주세요. DACON 제출용이 아닙니다.\n']
    notebook['cells'][2]['source'] = ['## 1. separation_bundle.zip 업로드\n']
    upload = ''.join(notebook['cells'][3]['source'])
    old = upload.split('hexdigest() == "')[1].split('"')[0]
    upload = upload.replace(old,hashlib.sha256(output.read_bytes()).hexdigest()).replace('mixed_probe_bundle.zip','separation_bundle.zip')
    upload = upload.replace('학습 실험 음원 무결성 확인 완료','분리 비교 음원 무결성 확인 완료')
    notebook['cells'][3]['source'] = upload.splitlines(True)
    notebook['cells'][-2]['source'] = ['## 4. 고정 가중치로 분리 전후 비교\n',
        '진단을 위해 모든 입력을 분리합니다. 실제 제출 코드에서 분리를 생략했을지 여부도 기록합니다.\n',
        '분리 음악의 음량 보정은 고정 규칙입니다. 무음 기준보다 작은 신호는 증폭하지 않고 기록합니다.\n',
        '분리 음악 WAV, 특징, 점수, 가중치, 실행 코드와 로그를 결과 ZIP에 저장합니다.\n']
    runner = ''.join(notebook['cells'][-1]['source']).replace('run_mixed_probe.py','run_separation_check.py').replace('mixed_probe_results','separation_results')
    notebook['cells'][-1]['source'] = runner.splitlines(True)
    path = ROOT/'notebooks/colab_separation_check.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print('READY',len(rows),'inputs',output.stat().st_size,'bytes',hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':
    main()
