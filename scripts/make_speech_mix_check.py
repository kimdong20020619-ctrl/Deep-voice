"""Bundle eight public speech speakers and matched real/generated music controls."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEECH = ROOT / 'data/raw/real-speech-20260916'
speech_manifest = json.loads((SPEECH / 'manifest.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(ROOT / 'frozen_probe_bundle.zip') as base:
    frozen = base.read('frozen_probe.npz')
with zipfile.ZipFile(ROOT / 'file_probe_bundle.zip') as base:
    original = json.loads(base.read('probe/manifest.json'))
    music = [r for r in original['rows'] if r['split']=='development']
    music.sort(key=lambda r:(r['file_fake'],r['ID']))
    assert len(music)==16 and sum(r['file_fake'] for r in music)==8
    assert len(speech_manifest['rows'])==8 and len({r['speaker'] for r in speech_manifest['rows']})==8
    manifest = dict(purpose='real_speech_and_mixed_audio_regression_not_full_competition_validation',
                    probe_sha256=hashlib.sha256(frozen).hexdigest(), speech=speech_manifest['rows'],
                    speech_source=speech_manifest['source'], music=music, rows=music,
                    mixing='both sources RMS 0.05; voice/music -6 or +6 dB; shared peak guard 0.95; float WAV')
    out = ROOT / 'speech_mix_bundle.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for name in base.namelist():
            if not name.startswith('probe/') and name!='run_file_probe_experiment.py':
                z.writestr(name,base.read(name))
        for name in ('MUSAN_LICENSE.txt','MUSAN_ANNOTATIONS.txt','ATTRIBUTION.txt'):
            z.writestr('probe/'+name,base.read('probe/'+name))
        for row in music:
            name='probe/test/'+row['ID']+'.wav'
            z.writestr(name,base.read(name))
        for row in speech_manifest['rows']:
            blob=(SPEECH/row['local_path']).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==row['sha256']
            z.writestr('probe/speech/'+row['local_path'],blob)
        z.writestr('probe/SPEECH_ATTRIBUTION.txt','LibriSpeech ASR corpus, Vassil Panayotov et al., 2015. https://www.openslr.org/12/ . CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ . Source clips: manifest.json. Changes in Colab: centered 8s crops, RMS normalization, music mixing at two ratios, shared peak guard. Source archive full checksum not verified.\n')
        z.writestr('probe/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.writestr('frozen_probe.npz',frozen)
        for name in ('run_file_probe_experiment.py','run_speech_mix_check.py'):
            z.write(ROOT/'scripts'/name,name)
nb=json.loads((ROOT/'notebooks/colab_frozen_probe.ipynb').read_text(encoding='utf-8'))
nb['cells'][0]['source']=['# 실제 음성·혼합 파일 회귀 검사\n',
    '고정 가중치로 실제 음성 8개, 음악 대조 16개, 혼합 32개를 비교합니다. 새로운 독립 음원 56개라는 뜻은 아닙니다.\n',
    '음성은 새 LibriSpeech 화자 8명, 음악은 기존 개발 원본입니다. 모델 사전학습과의 중복은 확인하지 못했습니다.\n',
    '생성 음성은 포함하지 않습니다. Voice EER·성분 지표·전체 대회 Score는 계산하지 않습니다.\n',
    '실제 음성만 있는 그룹에는 EER/AUC 대신 점수와 0.5 기준 가짜 판정 비율을 기록합니다.\n',
    'speech_mix_bundle.zip을 올리고 결과 speech_mix_results.zip을 보내주세요. 재학습·자동 제출은 하지 않습니다.\n']
nb['cells'][2]['source']=['## 1. speech_mix_bundle.zip 업로드\n']
upload=''.join(nb['cells'][3]['source'])
old=upload.split('hexdigest() == "')[1].split('"')[0]
upload=upload.replace(old,hashlib.sha256(out.read_bytes()).hexdigest()).replace('frozen_probe_bundle.zip','speech_mix_bundle.zip')
upload+='\nfor row in manifest["speech"]:\n    path = WORK / "probe/speech" / row["local_path"]\n    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]\nprint("실제 음성 원본 해시 확인 완료; 전체 디코딩은 마지막 셀에서 검사합니다.")\n'
nb['cells'][3]['source']=upload.splitlines(True)
nb['cells'][-2]['source']=['## 4. 음성 디코딩·혼합 생성·고정 판별기 비교\n',
    '여기서 FLAC 전체 디코딩과 길이/채널/유한값 검사를 수행합니다. 로컬에서는 FLAC 헤더와 해시만 확인했습니다.\n',
    '두 혼합 조건은 음성 RMS가 음악 대비 -6dB 또는 +6dB입니다. 가짜 음악이 섞이면 파일 정답은 FAKE=1입니다.\n',
    '생성 음악에 가창이 들어 있을 수 있어 성분별 가짜/존재 정답을 추측하지 않습니다.\n']
runner=''.join(nb['cells'][-1]['source']).replace('run_frozen_probe_check.py','run_speech_mix_check.py').replace('frozen_probe_results','speech_mix_results')
nb['cells'][-1]['source']=runner.splitlines(True)
target=ROOT/'notebooks/colab_speech_mix.ipynb'
target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('READY',target,'bundle bytes',out.stat().st_size,'sha256',hashlib.sha256(out.read_bytes()).hexdigest())
