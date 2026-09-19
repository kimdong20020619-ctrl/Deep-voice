"""Package a frozen external comparison with original 60-second audio and source metadata."""
import hashlib
import json
from pathlib import Path
import zipfile

from run_external_music_check import validate_plan
from collect_external_music import describe

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/raw/ai-openbmat-20260917'


def main():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    validate_plan(manifest)
    assert len(manifest['rows'])==48
    for pair in {row['pair_id'] for row in manifest['rows']}:
        ai=json.loads((DATA/'metadata'/(pair+'_ai.json')).read_text(encoding='utf-8'))
        human=json.loads((DATA/'metadata'/(pair+'_human.json')).read_text(encoding='utf-8'))
        assert describe(ai)==describe(human), 'Unmatched source metadata: '+pair
        assert ai['artisan']=='ai' and human['artisan']=='human'
        assert ai['reference_structure']['file']==human['reference_structure']['file']
    previous=ROOT/'data/experiments/mixed-probe-20260916/mixed_probe_results.zip'
    assert hashlib.sha256(previous.read_bytes()).hexdigest()=='4730fc617099dc423eb17bc9b230474d100490951127a54236f435214addebf1'
    with zipfile.ZipFile(previous) as result:
        weights=dict(music_probe=result.read('previous_music_probe.npz'),mixed_probe=result.read('experimental_mixed_probe.npz'))
    manifest['model_sha256']={name:hashlib.sha256(blob).hexdigest() for name,blob in weights.items()}
    manifest['inference']='Native full-minute inputs decoded/resampled by unchanged submission loader; max segment aggregation; fixed weights; no normalization fitting or model selection'
    output=ROOT/'external_music_bundle.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z,zipfile.ZipFile(ROOT/'separation_bundle.zip') as base:
        for name in base.namelist():
            if name.startswith(('model/','src/')) or name in ('script.py','requirements.txt'):
                z.writestr(name,base.read(name))
        for row in manifest['rows']:
            blob=(DATA/'audio'/(row['ID']+'.wav')).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==row['clip_sha256']
            z.writestr('probe/test/'+row['ID']+'.wav',blob)
            meta=(DATA/'metadata'/(row['ID']+'.json')).read_bytes()
            assert hashlib.sha256(meta).hexdigest()==row['metadata_sha256']
            z.writestr('probe/metadata/'+row['ID']+'.json',meta)
        z.writestr('probe/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        z.write(DATA/'record.json','probe/record.json')
        z.write(DATA/'selection_rejections.json','probe/selection_rejections.json')
        z.writestr('probe/ATTRIBUTION.txt',
            'AI-OpenBMAT (2026), David Lopez-Ayala, Asier Cabello, Pablo Zinemanas, Emilio Molina, Martin Rocamora. '
            'Official dataset: https://zenodo.org/records/20031232 . Project: https://github.com/DaveLoay/AI-OpenBMAT . '
            'Paper DOI: 10.1109/ICASSP55912.2026.11464623 . Release license: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/ . '
            'Source audio bytes unchanged; local filenames changed. Inference downmixes/resamples to 16kHz using the existing loader. '
            'Exact source paths and metadata hashes in manifest.json; published record metadata in record.json. '
            'No full archive checksum verification. These are new project inputs, not proven absent from foundation-model pretraining.\n')
        for name,blob in weights.items():
            z.writestr(name+'.npz',blob)
        for name in ('run_external_music_check.py','run_file_probe_experiment.py'):
            z.write(ROOT/'scripts'/name,name)
    nb=json.loads((ROOT/'notebooks/colab_separation_check.ipynb').read_text(encoding='utf-8'))
    nb['cells'][0]['source']=['# 새로운 방송 음원에서 고정 모델 비교\n',
        'T4 GPU를 선택하고 모두 실행한 뒤 external_music_bundle.zip을 업로드하세요.\n',
        '새 원본 24쌍(사람 음악 24개 + Suno 음악 24개)의 60초 파일을 평가합니다. 기존 세 모델은 고정하며 재학습하지 않습니다.\n',
        'ai 폴더의 무음악 파일은 제외했습니다. 음악·음성 원본이 쌍 사이에 겹치지 않는 표본입니다.\n',
        '공식 전체 벤치마크가 아닌 외부 표본 검사이며 대회 Score가 아닙니다. 결과 external_music_results.zip을 보내주세요.\n']
    nb['cells'][2]['source']=['## 1. external_music_bundle.zip 업로드\n']
    upload=''.join(nb['cells'][3]['source'])
    old=upload.split('hexdigest() == "')[1].split('"')[0]
    upload=upload.replace(old,hashlib.sha256(output.read_bytes()).hexdigest()).replace('separation_bundle.zip','external_music_bundle.zip').replace('분리 비교 음원','외부 검증 음원')
    nb['cells'][3]['source']=upload.splitlines(True)
    nb['cells'][-2]['source']=['## 4. 고정 모델 외부 평가\n',
        '원본 60초 전체를 사용합니다. 모델 입력만 기존 로더로 16kHz 변환합니다. 음악 분리나 추가 음량 보정은 하지 않습니다.\n',
        '파일별 예측·특징·고정 가중치·코드·메타데이터·로그를 보관합니다. 파일 단위 EER와 AUC만 계산합니다.\n']
    runner=''.join(nb['cells'][-1]['source']).replace('run_separation_check.py','run_external_music_check.py').replace('separation_results','external_music_results')
    nb['cells'][-1]['source']=runner.splitlines(True)
    target=ROOT/'notebooks/colab_external_music.ipynb'
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('READY',len(manifest['rows']),output.stat().st_size,hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__':
    main()
