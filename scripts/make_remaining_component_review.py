"""Freeze ten existing-source clips for presence review, without consulting predictions."""
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/component-remainder-20260920'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def order(row):
    return sha(('component-remainder-v1:' + row['ID']).encode())


def main():
    speech_dir = ROOT / 'data/raw/paired-speech-20260916'
    music_dir = ROOT / 'data/raw/file-probe-20260915'
    speech = json.loads((speech_dir / 'manifest.json').read_text(encoding='utf-8'))['rows']
    music = json.loads((music_dir / 'manifest.json').read_text(encoding='utf-8'))['rows']
    selected = []
    for generator in sorted({r['generator'] for r in music if r['file_fake']}):
        row = min((r for r in music if r.get('generator') == generator), key=order)
        selected.append(('music', row))
    assert len(selected) == 5
    for split, generators in [('train', ['human', 'ljspeech_hifiGAN', 'ljspeech_melgan']),
                              ('development', ['human', 'ljspeech_waveglow'])]:
        real = min((r for r in speech if r['split'] == split and r['file_fake'] == 0), key=order)
        for generator in generators:
            matches = [r for r in speech if r['utterance'] == real['utterance'] and r['generator'] == generator]
            assert len(matches) == 1
            selected.append(('speech', matches[0]))
    assert len(selected) == 10
    (OUT / 'clips').mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (kind, original) in enumerate(selected, 1):
        identity = f'C{i:02d}'
        if kind == 'music':
            blob = (music_dir / 'test' / (original['ID']+'.wav')).read_bytes()
            assert sha(blob) == original['clip_sha256']
            transform = 'Existing eight-second diagnostic clip; bytes unchanged.'
        else:
            source_blob = (speech_dir / original['local_path']).read_bytes()
            assert sha(source_blob) == original['source_sha256']
            sr, pcm = wavfile.read(io.BytesIO(source_blob))
            assert pcm.ndim == 1 and pcm.dtype == np.int16
            divisor = math.gcd(sr, 16000)
            audio = resample_poly(pcm.astype(np.float64)/32768, 16000//divisor, sr//divisor)
            assert len(audio) >= 128000
            start = (len(audio)-128000)//2
            clip = audio[start:start+128000]
            gain = min(1.0, 0.95/max(float(np.max(np.abs(clip))), 1e-12))
            output = io.BytesIO()
            wavfile.write(output, 16000, np.rint(clip*gain*32768).astype(np.int16))
            blob = output.getvalue()
            transform = dict(method='scipy.signal.resample_poly, center crop, peak protection only',
                             original_sample_rate=sr, start_sample_16k=start, gain=gain)
        sr, pcm = wavfile.read(io.BytesIO(blob))
        assert sr == 16000 and pcm.shape == (128000,) and pcm.dtype == np.int16 and np.any(pcm)
        (OUT / 'clips' / (identity+'.wav')).write_bytes(blob)
        rows.append(dict(ID=identity, source_kind=kind, original=original,
            clip_sha256=sha(blob), pcm_sha256=sha(pcm.tobytes()), transform=transform,
            file_fake=original['file_fake'], voice_presence=None, music_presence=None,
            voice_fake=None, music_fake=None, component_labels_reviewed=False,
            license='CC BY-NC 4.0' if kind == 'music' else original['license'],
            source_previously_used=True, independent_holdout=False))
    assert len({r['clip_sha256'] for r in rows}) == len({r['pcm_sha256'] for r in rows}) == 10
    result = dict(rows=rows,
        selection='Fixed SHA256 order within generator or matched utterance, no predictions consulted.',
        purpose='Presence-label review for tiny diagnostic only; not training or independent validation.',
        limitations=['All selected sources were used previously; original split names do not define a new holdout.',
            'Five speech clips share one reference speaker and only two utterance groups.',
            'One music example per generator cannot establish generator generalization.',
            'Authenticity inherited from source provenance; presence labels await user review.'],
        ai_usage_log=dict(date='2026-09-20', tool='Codex/Python',
            action='Prepared fixed source-linked clips for user component review; no inference or training.'))
    manifest = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    (OUT / 'manifest.json').write_text(manifest, encoding='utf-8')
    instructions = '''성분 확인용 음원 10개, 각 8초입니다. Colab은 필요 없습니다.
clips/C01.wav부터 C10.wav까지 듣고 각 파일을 다음 중 하나로 알려주세요.
1. 악기 음악만 들림
2. 사람 목소리만 들림 (말, 노래, 허밍 모두 포함)
3. 악기 음악과 사람 목소리가 모두 들림
4. 둘 다 없거나 잘 모르겠음 (가능하면 설명)

예: C01 음악만, C02 둘 다, C03 잘 모르겠음
채팅으로 답하면 됩니다. review.csv를 직접 적을 필요는 없습니다.
진짜인지 가짜인지 맞히는 검사가 아닙니다. 들리는 성분만 확인해주세요.
이미 확인한 이전 음악 9개는 다시 들을 필요가 없습니다.
이번 자료는 기존 실험에 사용한 원본입니다. 독립 검증셋이 아니며 점수 향상 근거로 쓸 수 없습니다.
모든 음원은 개별 원본을 자른 것으로, 서로 합성해 섞지 않았습니다.
manifest.json과 ATTRIBUTION.txt 및 라이선스 문서를 함께 보관하세요.
'''
    attribution = '''FakeMusicCaps: https://zenodo.org/records/15063698, CC BY-NC 4.0.
Five existing eight-second crops. Original paths, generator names and hashes are in manifest.json.
LJSpeech: Keith Ito and Linda Johnson, 2017, https://keithito.com/LJ-Speech-Dataset/, Public Domain.
WaveFake: Joel Frank and Lea Schoenherr, 2021, https://zenodo.org/records/5642694, CC BY-SA 4.0.
Speech changes: 22050-to-16000Hz resampling, center eight-second crop, peak protection if needed.
WaveFake crops remain CC BY-SA 4.0. FakeMusicCaps crops remain CC BY-NC 4.0.
This is a collection of separately licensed files; no common license overrides source terms.
CC BY-NC 4.0: https://creativecommons.org/licenses/by-nc/4.0/
CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
'''
    sheet = io.StringIO(newline='')
    writer = csv.writer(sheet)
    writer.writerow(['ID', 'voice_present', 'instrument_music_present', 'uncertain', 'notes'])
    writer.writerows([r['ID'], '', '', '', ''] for r in rows)
    target = Path.home() / 'Downloads/component_remaining_review.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('읽어주세요.txt', instructions.encode('utf-8-sig'))
        archive.writestr('ATTRIBUTION.txt', attribution)
        archive.writestr('manifest.json', manifest)
        archive.writestr('review.csv', sheet.getvalue().encode('utf-8-sig'))
        archive.write(speech_dir / 'WaveFake_LICENSE.txt', 'WaveFake_LICENSE.txt')
        for row in rows:
            archive.write(OUT / 'clips' / (row['ID']+'.wav'), 'clips/'+row['ID']+'.wav')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for row in rows:
            assert sha(archive.read('clips/'+row['ID']+'.wav')) == row['clip_sha256']
    print(target)
    print('bytes', target.stat().st_size, 'sha256', sha(target.read_bytes()))
    print('PASS: source hashes, 10 unique eight-second 16k PCM clips, ZIP CRC and member hashes')


if __name__ == '__main__':
    main()
