"""Build a development-only mixed-speech training experiment and Colab bundle."""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from run_speech_mix_check import mix, safe_level

ROOT = Path(__file__).resolve().parents[1]
SPEECH = ROOT/'data/raw/paired-speech-20260916'


def speech_crop(blob):
    sr, pcm = wavfile.read(io.BytesIO(blob))
    assert pcm.ndim == 1 and pcm.dtype == np.int16 and len(pcm) >= 8*sr
    divisor = math.gcd(sr, 16000)
    audio = resample_poly(pcm.astype(np.float64)/32768, 16000//divisor, sr//divisor)
    start = (len(audio)-128000)//2
    return safe_level(audio[start:start+128000])


def main():
    speech_manifest = json.loads((SPEECH/'manifest.json').read_text(encoding='utf-8'))
    sources, waves = [], {}
    with zipfile.ZipFile(ROOT/'file_probe_bundle.zip') as original:
        music_manifest = json.loads(original.read('probe/manifest.json'))
        license_text = original.read('probe/MUSAN_LICENSE.txt').decode()
        for row in music_manifest['rows']:
            if row['split'] == 'holdout':
                continue
            blob = original.read('probe/test/'+row['ID']+'.wav')
            assert hashlib.sha256(blob).hexdigest() == row['clip_sha256']
            sr, pcm = wavfile.read(io.BytesIO(blob))
            assert sr == 16000 and pcm.dtype == np.int16
            waves[row['ID']] = safe_level(pcm.astype(float)/32768)
            license_block = ''
            if not row['file_fake']:
                matches = [b for b in re.split(r'={10,}', license_text) if Path(row['source']).stem in b]
                assert len(matches) == 1
                license_block = matches[0]
            # Only unambiguous BY 4.0 / BY-SA 4.0 can enter a WaveFake-SA mixture.
            sa_compatible = ('CC BY 4.0' in license_block or 'CC BY-SA 4.0' in license_block) and 'NC' not in license_block and 'ND' not in license_block
            sources.append(dict(row, kind='music', source_group='music:'+str(row['file_fake'])+':'+row['group'],
                                sa_compatible=sa_compatible, license_evidence=license_block if license_block else 'FakeMusicCaps CC BY-NC 4.0'))
        for row in speech_manifest['rows']:
            blob = (SPEECH/row['local_path']).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == row['source_sha256']
            waves[row['ID']] = speech_crop(blob)
            sources.append(dict(row, kind='speech', source_group='utterance:'+row['utterance']))
        rows, audio_files = [], {}

        def add(identity, split, family, selected, audio, db=None):
            licensed_sa = any(s.get('license') == 'CC BY-SA 4.0' for s in selected)
            music_source = next((s for s in selected if s['kind']=='music'), None)
            if licensed_sa:
                derivative_license = 'CC BY-SA 4.0'
            elif music_source is None:
                derivative_license = 'Public Domain source; preparation contributions CC0'
            elif music_source['file_fake']:
                derivative_license = 'CC BY-NC 4.0'
            else:
                derivative_license = 'Same license as the music source in license_evidence; preparation contributions offered under those terms'
            output = io.BytesIO()
            wavfile.write(output, 16000, audio.astype(np.float32))
            blob = output.getvalue()
            audio_files[identity] = blob
            rows.append(dict(ID=identity, split=split, family=family,
                             file_fake=max(s['file_fake'] for s in selected),
                             source_ids=[s['ID'] for s in selected],
                             source_groups=[s['source_group'] for s in selected],
                             source_hashes=[s['source_sha256'] for s in selected],
                             clip_sha256=hashlib.sha256(blob).hexdigest(), voice_to_music_db=db,
                             derivative_license=derivative_license))

        for source in sources:
            add(source['ID'], source['split'], source['kind'], [source], waves[source['ID']])
        for split in ('train', 'development'):
            speech = [s for s in sources if s['kind']=='speech' and s['split']==split]
            music_rows = [s for s in sources if s['kind']=='music' and s['split']==split]
            real_voices = [s for s in speech if not s['file_fake']]
            for i, music in enumerate(music_rows):
                voice = real_voices[i % len(real_voices)]
                for db in (-6, 6):
                    audio, _ = mix(waves[voice['ID']], waves[music['ID']], db)
                    add(f'X_{music["ID"]}_0_{db:+d}', split, f'mix_voice_{db:+d}dB', [voice, music], audio, db)
            compatible_music = [s for s in music_rows if not s['file_fake'] and s['sa_compatible']]
            assert compatible_music, 'No licensed real music for fake speech mixtures'
            for i, voice in enumerate(s for s in speech if s['file_fake']):
                music = compatible_music[i % len(compatible_music)]
                for db in (-6, 6):
                    audio, _ = mix(waves[voice['ID']], waves[music['ID']], db)
                    add(f'XF_{voice["ID"]}_{db:+d}', split, f'mix_voice_{db:+d}dB', [voice, music], audio, db)
        manifest = dict(rows=rows, sources=sources,
            purpose='Mixed training and development selection only; no independent holdout or competition Score',
            limitations=[speech_manifest['limitation'], 'Development music was previously used for model selection.',
                        'No fake-voice plus fake-music mixtures: differing source license requirements.',
                        'Speech pretrained overlap unknown; component labels not inferred from music.'])
        from run_mixed_probe import validate_manifest
        validate_manifest(manifest)
        out = ROOT/'mixed_probe_bundle.zip'
        with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in original.namelist():
                if not name.startswith('probe/') and name != 'run_file_probe_experiment.py':
                    z.writestr(name, original.read(name))
            for name in ('MUSAN_LICENSE.txt','MUSAN_ANNOTATIONS.txt','ATTRIBUTION.txt'):
                z.writestr('probe/'+name, original.read('probe/'+name))
            z.write(SPEECH/'WaveFake_LICENSE.txt', 'probe/WaveFake_LICENSE.txt')
            attribution = ('WaveFake: Joel Frank and Lea Schoenherr, 2021, https://zenodo.org/records/5642694, CC BY-SA 4.0. '
                'LJSpeech: Keith Ito and Linda Johnson, 2017, https://keithito.com/LJ-Speech-Dataset/, Public Domain. '
                'Changes: resampling to 16kHz, centered 8s crop, RMS 0.05 normalization, optional mixing and shared peak limit 0.95. '
                'WaveFake derived audio is CC BY-SA 4.0; source IDs and real-music credits are in manifest.json and MUSAN_LICENSE.txt. '
                'Other audio retains its source-specific terms. This collection has no blanket license.\n')
            z.writestr('probe/SPEECH_ATTRIBUTION.txt', attribution)
            z.writestr('probe/manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
            for identity, blob in audio_files.items():
                z.writestr('probe/test/'+identity+'.wav', blob)
            with zipfile.ZipFile(ROOT/'frozen_probe_bundle.zip') as frozen:
                z.writestr('previous_music_probe.npz', frozen.read('frozen_probe.npz'))
            for name in ('run_file_probe_experiment.py', 'run_mixed_probe.py'):
                z.write(ROOT/'scripts'/name, name)
    nb = json.loads((ROOT/'notebooks/colab_file_probe.ipynb').read_text(encoding='utf-8'))
    nb['cells'][0]['source'] = ['# 음성·음악 혼합 학습 실험\n',
        '기존 음악 전용 판별기와 새 혼합 학습 판별기를 비교합니다. 독립 최종 검증이나 제출용이 아닙니다.\n',
        '실제/생성 음성의 문장과 생성기는 학습·개발에서 분리했지만 기준 화자는 같습니다. 음악은 기존 학습·개발 자료입니다.\n',
        'T4 GPU를 선택하고 모두 실행한 뒤 mixed_probe_bundle.zip을 업로드하세요. 결과 mixed_probe_results.zip을 보내주세요.\n']
    nb['cells'][2]['source'] = ['## 1. mixed_probe_bundle.zip 업로드\n']
    upload = ''.join(nb['cells'][3]['source'])
    old = upload.split('hexdigest() == "')[1].split('"')[0]
    upload = upload.replace(old, hashlib.sha256(out.read_bytes()).hexdigest()).replace('file_probe_bundle.zip', 'mixed_probe_bundle.zip')
    nb['cells'][3]['source'] = upload.splitlines(True)
    nb['cells'][-2]['source'] = ['## 혼합 특징 추출 → 학습 → 개발 비교\n',
        '기존 임베딩 위의 작은 판별기만 학습합니다. C 세 후보를 개발 조건별 평균 EER로 선택합니다.\n',
        '홀드아웃·대회 데이터는 사용하지 않습니다. 전처리·학습 코드와 특징·예측·환경을 결과 ZIP에 보관합니다.\n']
    runner = ''.join(nb['cells'][-1]['source']).replace('run_file_probe_experiment.py', 'run_mixed_probe.py').replace('probe_results', 'mixed_probe_results').replace('file_mixed_probe_results', 'mixed_probe_results')
    nb['cells'][-1]['source'] = runner.splitlines(True)
    target = ROOT/'notebooks/colab_mixed_probe.ipynb'
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('READY',len(rows), {s:sum(r['split']==s for r in rows) for s in ('train','development')})
    print('SHA256',hashlib.sha256(out.read_bytes()).hexdigest(), 'bytes',out.stat().st_size)


if __name__ == '__main__':
    main()
