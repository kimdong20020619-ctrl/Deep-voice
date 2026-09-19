# 고정 모델의 새 외부 음원 검증 준비 — 2026-09-17

## 판단과 목적

기존 작은 검증셋에서 반복한 혼합 학습과 음악 분리는 제출 모델 교체를 뒷받침하지 못했다. 이전 판단은 `26_mixed-probe` 및 `28_separation` 관련 기록과 결과 원본을 기준으로 한다. 이번에는 가중치를 변경하지 않고 새 데이터 출처에서 기본 DF-Arena, 음악 전용 판별기, 혼합 학습 판별기를 비교한다. GPU 결과가 나오기 전에는 개선이나 제출 적합성을 주장하지 않는다.

이 문서는 내부 실험 기록이다. 대회 제출 보고서 본문이 아니다.

## 공식 출처와 사용 범위

- 데이터: https://zenodo.org/records/20031232
- 확인한 API 원문: https://zenodo.org/api/records/20031232
- 프로젝트: https://github.com/DaveLoay/AI-OpenBMAT
- 논문: https://arxiv.org/html/2602.06823v1
- 논문 DOI: https://doi.org/10.1109/ICASSP55912.2026.11464623
- 배포 라이선스: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- 저자: David López-Ayala, Asier Cabello, Pablo Zinemanas, Emilio Molina, Martín Rocamora.

공식 메타데이터와 프로젝트 원문을 확인했다. 레코드 공개일은 2026-05-05다. AI-OpenBMAT는 사람 음악과 Suno v3.5 음악을 방송 음성과 조합한 데이터다. 원문 API는 `data/raw/ai-openbmat-20260917/record.json`에 보관했다. 원본 파일명, WAV/JSON SHA-256, 출처 및 표시 의무를 번들에 포함했다. WAV 바이트는 그대로 보존하고 로컬 이름만 변경했다. 추론 입력은 기존 제출 로더가 16 kHz로 변환한다.

전체 ZIP은 7,312,205,691바이트이며 게시된 MD5는 `436f1bcc46e9455fc4f732205b8dcb99`다. 필요한 파일만 HTTP Range로 받았으므로 **전체 ZIP MD5는 미검증**이다. 개별 ZIP 항목 CRC와 받은 파일 SHA-256을 확인했다.

## 표본 선정과 누수 확인

`scripts/collect_external_music.py`의 고정 SHA-256 순서로 첫 적격 24쌍을 선정했다. 예측 점수는 선정에 사용하지 않았다. 사람 음악 24개와 AI 음악 24개, 총 48개이며 실제 WAV 길이는 모두 60초, 22,050 Hz, 모노 PCM16이다. 근거는 로컬 `manifest.json`의 샘플 수와 WAV 직접 디코딩 결과다.

- 음악 주석 구간 합계가 8초 미만인 후보 14개를 제외했다. `ai/` 폴더에도 음악 없는 파일이 있으므로 폴더명만으로 FAKE를 붙이지 않았다.
- 앞서 선택한 음악 또는 음성 원본과 겹치는 후보 2개를 제외했다.
- 각 쌍의 이벤트 종류·구간 길이·음성 파일명·시작 위치·지속 시간, 음악 원본 파일명 목록, 방송 참조 ID가 일치한다. 음악 파일명 일치는 원본 ID 대응이며 사람/AI WAV가 같다는 의미가 아니다.
- 서로 다른 쌍 사이에서 음악 및 음성 원본 토큰이 겹치지 않는다. 음악 토큰은 사람/AI 구분을 포함한다.
- `music_seconds`는 주석 구간 길이의 합이다. 교차 페이드 때문에 60초를 약간 넘는 값이 있으며, 실제 비중복 가청 음악 길이로 해석하지 않는다.
- 이전 음악 파일럿 및 음악 판별기 데이터의 기록된 SHA-256과 새 WAV의 정확한 해시 중복은 없다. 근접 중복과 기초 모델 사전학습 데이터 중복은 미검증이다. 서로 다른 원본 ID가 같은 가수·음성 화자라는 가능성도 배제하지 않는다.
- 공식 전체 벤치마크가 아닌 음악 존재 조건으로 선정한 작은 외부 표본이다. 대회 데이터 대표성을 보장하지 않는다.

## 고정 모델과 평가

기존 결과 `data/experiments/mixed-probe-20260916/mixed_probe_results.zip`에서 가중치를 가져왔다. 원본 결과 ZIP SHA-256은 `4730fc617099dc423eb17bc9b230474d100490951127a54236f435214addebf1`이다.

- 음악 판별기: `08bfc83eab9f4dd23a4f3c9c884f0f1c9e25230634ff4245da8d3a289539601f`
- 혼합 판별기: `8b6b6d617d7f096342a1bc75fc8587390ceb7ccafb86ffe129a89243b6912f1b`

기본 모델과 두 판별기는 가중치·표준화 계수·집계 방식(max)을 고정한다. 추가 학습, 임계값 최적화, 음악 분리, 추가 음량 보정은 하지 않는다. 원본 60초 전체에 기존 파일 로더와 구간 집계를 적용한다. 8초 실험과 입력 길이가 다르므로 차이를 생성기 하나의 영향으로 단정하면 안 된다.

파일 EER, 파일 ROC-AUC, 쌍별 AI 점수가 사람 점수보다 높은 횟수 및 동률 수를 기록한다. FAKE=1이며 EER은 제공된 공식 구현을 사용한다. Voice/Music EER 및 Presence AUC는 계산하지 않는다. 따라서 출력값은 **대회 총점 Score가 아니다**. 결과로 자동 모델 선택·배포·DACON 제출을 수행하지 않는다.

## 산출물과 검증

- `external_music_bundle.zip`: 108,900,667바이트
- SHA-256: `ecf54cd5945117dfe798b1e383aaabb0b583a143c431da41d1c019930c26ae28`
- `notebooks/colab_external_music.ipynb`
- 실행 결과 이름: `external_music_results.zip`
- 준비·수집·패키징: `scripts/prepare_external_music.py`, `collect_external_music.py`, `make_external_music_check.py`
- 평가: `scripts/run_external_music_check.py`
- 검증: `python -X utf8 -B scripts/selftest_external_music.py` — 3개 모두 통과, 건너뜀 없음.

검사 범위는 잘못된 라벨/분할/중복 원본 거부, 실제 LinearProbe 코드와 모의 GPU를 통한 결과 생성 및 가중치 보존, 완성 번들의 CRC/파일 SHA/음원 형식/노트북 코드 문법/실행 코드 일치다. 번들 생성 시 전체 쌍의 원본 메타데이터도 재검사했다. 로컬에는 실제 GPU 추론 의존성이 없어 **실제 DF-Arena GPU 실행, 실행 시간, 점수, L4 제출 서버 재현은 미검증**이다.

결과 ZIP에는 예측, 임베딩, 고정 판별기 가중치, 실행 코드, 환경·실행 로그, 출처 메타데이터가 포함된다. 반환 후 이 자료로 점수 재계산과 가중치 일치를 검증한다.

## 사용자 실행 순서

1. Colab에서 다운로드 폴더의 `colab_external_music.ipynb`를 연다.
2. `런타임 → 런타임 유형 변경 → T4 GPU → 저장`을 선택한다.
3. `런타임 → 모두 실행`을 선택한다.
4. 파일 선택 창에서 `external_music_bundle.zip` 하나를 선택한다.
5. 완료되면 내려받는 `external_music_results.zip`을 PC 다운로드 폴더에 둔다.

실패해도 마지막 실행 셀은 가능한 로그 ZIP을 내려받게 한다. 오류가 있는 결과를 성공으로 취급하지 않는다. 모델 다운로드에는 인터넷이 필요하며, 이 실험 노트북은 DACON 제출 ZIP이 아니다.

## AI 활용 기록

| 날짜 | 도구 | 작업 | 사람 확인/미검증 |
|---|---|---|---|
| 2026-09-17 | Codex, 로컬 Python/PowerShell, 공식 웹/API | 새 외부 데이터 선정·출처 확인, 고정 비교 코드·Colab 번들 제작, 로컬 검사 | 사용자가 Colab 실행 후 결과 전달 예정. 실제 GPU 성능 및 대회 점수 미검증 |
