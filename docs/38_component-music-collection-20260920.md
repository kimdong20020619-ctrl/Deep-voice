# 성분 검토용 음악 수집 결과 — 2026-09-20

## 완료한 작업

이전 감사에서 선정한 12곡 중 3곡을 보류하고 9곡의 공개 미러 WAV를 확보했다. 각 파일의 중앙 8초를 추출했으며 음량 재조정·소스 분리·모델 예측은 하지 않았다. 현재 성분 정답은 비어 있다. 실제 구간의 청취 확인이 다음 단계다.

- 수집 기록: [manifest.json](../data/raw/component-music-20260920/manifest.json)
- 수집 코드: [collect_component_music.py](../scripts/collect_component_music.py)
- 검토 묶음 생성: [make_component_review.py](../scripts/make_component_review.py)
- 사용자 파일: `C:\Users\kimdo\Downloads\component_music_review.zip`
- ZIP 크기: 2,179,533 bytes. SHA-256: `e9215e84818de47eaee8a7ed7063320886fe0715f00f4403008ca9f09fca929e`.

검토 ZIP에는 WAV 9개, 빈 review.csv, 한국어 설명, 출처·해시 manifest, 원본 주석과 라이선스 문서를 넣었다. 곡 전체 파일은 저장소의 `data/raw/component-music-20260920/originals`에 보관한다. 전체 파일 크기 합계는 86,104,862 bytes다. 이 경로의 'originals'는 수집한 WAV 전체를 뜻하며 공식 압축본의 바이트 원본임을 보장하는 이름이 아니다.

## 출처와 한계

[MUSAN 공식 페이지](https://www.openslr.org/17/)에서 배포 출처를 확인했다. 공식 서버 연결 중단과 스트림 전송 지연으로 개별 다운로드가 가능한 [shadwl/musan 공개 미러](https://huggingface.co/datasets/shadwl/musan)를 사용했다. 실제 받은 것은 Dataset Viewer의 WAV 내보내기이며, 응답 URL의 리비전 `4659edba2da70ebb8b9eccf6c997e7199a685119`를 코드에서 검사했다. 공식 압축본의 바이트·PCM과의 대조는 미수행이다. 전체 tar.gz 체크섬도 검증하지 않았다.

미러의 경로 식별자를 기존 MUSAN 주석·곡별 라이선스에 연결했다. 원본 주석 문서는 이전 보관 SHA-256과 대조 후 복사했다. 미러 출처를 공식 원본이라고 바꾸어 기록하지 않는다. `source_sha256`은 실제 내려받은 WAV에 대한 SHA-256이다.

MUSAN은 2015년 배포 자료다. 최신 모델 성능을 입증하는 데이터가 아니며, 이번에는 사람 제작 음악의 성분 확인용 소규모 후보로만 사용한다. Dataset Viewer 출력이 원본과 동일하다는 독립 검증은 아직 없으므로 최종 독립 검증셋이나 공식 대회 데이터와 동등하게 취급하지 않는다.

## 제외 및 수정

| 항목 | 수치/처리 | 출처 URL / 로컬 근거 | 기준연도 | 비고 |
|---|---|---|---|---|
| music-jamendo-0070 | 제외 | 저장된 `08_LICENSE.txt` | 2015 자료, 2026 확인 | 설명 4.0과 괄호 3.0이 충돌 |
| music-jamendo-0108 | 제외 | 주석·라이선스의 JPMOUNIER와 MOUNIER | 2015 자료, 2026 확인 | 동일인 가능성, 확정하지 않고 보류 |
| music-hd-0048 | 제외 | 저장된 `06_LICENSE.txt` | 2015 자료, 2026 확인 | Performer는 MIT, Source는 Bernd Krueger로 다른 후보와 관계 불명확 |
| music-hd-0002 | 라이선스 구체화 | 저장된 `06_LICENSE.txt` | 2015 자료, 2026 확인 | `CC BY-SA 3.0 DE` 국가 표기를 보존 |
| 최종 수집 | 9곡, 각 8초 | 수집 manifest 및 검토 ZIP | 2026 | 총 청취 구간 72초, 모두 성분 미검토 |

기존 두 음악 manifest와 트랙/아티스트 문자열이 겹치지 않도록 한 조건은 유지했다. 저장된 제외 목록과 수집 WAV 해시의 일치도 없었다. 하지만 형식 변환은 바이트 해시를 바꿀 수 있으므로 이것을 모든 원본 중복이 없다는 증거로 쓰지 않는다. 아티스트 별칭 전체, 동일 녹음, 다른 과거 데이터 및 사전학습 데이터 중복은 미검증이다.

## 검증 결과

- 수집 파일 9개의 SHA-256을 독립적인 재읽기로 확인.
- 8초 구간 9개 모두 16kHz·모노·PCM16·128,000 samples 확인, 무음이 아님을 확인.
- 수집 WAV 해시, 구간 파일 해시, 구간 PCM 해시 각각 중복 없음.
- ZIP CRC 검사 통과, WAV 파일 수 9개 확인.
- 두 Python 파일 구문 검사 통과.
- 모델 추론, 정확도·EER 계산, 점수 향상 검증 및 제출은 하지 않음.

## 사용자가 할 일

1. 다운로드 폴더의 `component_music_review.zip`을 모두 압축 해제한다. Colab 실행은 필요 없다.
2. clips 폴더의 9개 WAV를 재생한다.
3. review.csv에서 목소리 존재, 악기 음악 존재, 불확실 여부를 각각 1/0으로 적는다. 말·노래·허밍은 모두 목소리에 포함한다. 애매한 경우 uncertain=1로 표시하고 억지로 정답을 만들지 않는다.
4. 작성한 CSV를 다운로드 폴더에 보관한 뒤 완료됐다고 알려준다. 파일 내용을 보고 구간 라벨과 사용 범위를 다시 검토한다.

파일의 REAL/FAKE를 귀로 판별해 달라는 요청은 아니다. 사람 제작 출처인 음악에서 사용 구간의 성분만 확인한다. 청취 결과가 와도 이 묶음은 REAL 음악뿐이므로 단독 Music EER·전체 대회 Score를 계산할 수 없다. 후속 혼합 실험에 쓰려면 가짜 음성/가짜 음악, 음성 독립 원본 및 평가 설계를 별도로 확보해야 한다.

## 재현

저장소에서 `python -X utf8 -B scripts/collect_component_music.py`, 이어서 `python -X utf8 -B scripts/make_component_review.py` 실행. 이미 확보한 전체 파일은 receipt의 SHA-256을 검사한 뒤 재사용한다. 새 패키지는 설치하지 않았다. 공개 Viewer가 동일 리비전을 제공하지 않으면 중단하며 최신 리비전으로 조용히 바꾸지 않는다.

사용: Python/PowerShell — 수집, 해시·오디오 형식·ZIP 검증. 웹 원문 — 공식 배포 출처 확인.

## AI 활용 기록

| 날짜 | 도구 | 수행 | 한계 |
|---|---|---|---|
| 2026-09-20 | Codex, Python/PowerShell, 웹 | 음원 수집·라이선스 대조·검토 ZIP 준비 | 원본 바이트 대조·청취·GPU 추론·성능 향상은 미검증 |
