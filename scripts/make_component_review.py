"""Package short listening clips with unanswered component review fields."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/component-music-20260920'


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    rows = manifest['rows']
    assert len(rows) == 9
    text = io.StringIO(newline='')
    columns = ['track_id', 'voice_present', 'instrument_music_present', 'uncertain', 'notes']
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    writer.writerows(dict(track_id=r['track_id']) for r in rows)
    instructions = '''음악 성분 확인 — 학습/추론 실행 파일이 아닙니다.

1. ZIP을 모두 압축 해제합니다. Colab은 필요 없습니다.
2. clips 폴더의 WAV 파일 9개를 차례로 재생합니다. 각 파일은 8초입니다.
3. review.csv에 다음과 같이 적습니다. 듣지 않은 칸은 비워 둡니다.
   voice_present: 말소리, 노래, 허밍 등 사람 목소리가 들리면 1, 없으면 0.
   instrument_music_present: 악기/전자음으로 된 음악이 들리면 1, 없으면 0.
   uncertain: 판단하기 어려우면 1, 명확하면 0.
   notes: 특이점이나 확신이 없는 이유를 자유롭게 적습니다.
4. 작성한 review.csv를 다운로드 폴더에 저장하고 완료했다고 알려주세요.

억지로 정답을 고르지 마세요. 불확실한 구간은 평가에서 제외할 수 있습니다.
원본 주석은 보컬 없음이지만, 실제 잘라낸 구간은 아직 청취 검토되지 않았습니다.
진짜/가짜는 귀로 판별할 항목이 아닙니다. 이 묶음은 MUSAN의 사람 제작 음악 자료입니다.
기존 두 음악 manifest와 트랙/아티스트 문자열이 겹치지 않도록 고른 후보입니다.
전체 과거 자료·다른 녹음·사전학습 자료와의 중복은 완전히 검증하지 않았습니다.
진짜 음악만 있으므로 이것만으로 Music EER이나 대회 총점을 계산할 수 없습니다.

licenses 폴더는 원본 주석과 곡별 사용 조건입니다. 재배포 시 함께 보존하세요.
manifest.json에는 아티스트, 라이선스, 출처, 자른 위치, SHA-256을 기록했습니다.
공개 미러의 고정 리비전에서 제공하는 WAV 내보내기 파일입니다.
공식 전체 압축본 원본과 바이트가 일치하는지는 미검증입니다.
'''
    target = Path.home() / 'Downloads/component_music_review.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('읽어주세요.txt', instructions.encode('utf-8-sig'))
        archive.writestr('review.csv', text.getvalue().encode('utf-8-sig'))
        archive.write(SOURCE / 'manifest.json', 'manifest.json')
        for row in rows:
            path = SOURCE / 'clips' / (row['track_id'] + '.wav')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['clip_sha256']
            archive.write(path, 'clips/' + path.name)
        for path in sorted((SOURCE / 'licenses').glob('*.txt')):
            archive.write(path, 'licenses/' + path.name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len([n for n in archive.namelist() if n.endswith('.wav')]) == 9
    print(target)
    print('bytes', target.stat().st_size, 'sha256', hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
