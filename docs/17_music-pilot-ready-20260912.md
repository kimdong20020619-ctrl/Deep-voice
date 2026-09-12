# 첫 실제 음악 FILE 진단 실험 — 실행 준비

## 준비된 파일

- `notebooks/colab_music_pilot.ipynb`: 무료 Colab에서 실행할 새 노트북.
- `music_pilot_bundle.zip`: 모델 코드·설정·라이선스·실제 음원 크롭을 포함하는 업로드 파일. 대회 제출용이 아니다.
- `data/raw/music-pilot-20260912/`: 원본 20개, 크롭, 정답, 출처·해시·크롭 위치 기록.
- 재생성: `scripts/prepare_music_pilot.py`, `scripts/make_music_pilot_notebook.py`.
- 로컬 검사: `scripts/selftest_music_pilot.py`.

이 문서는 내부 실험 기록이며 공모전 제출 보고서 본문이 아니다.

## 실제 데이터와 선정 근거

공식 MUSAN 압축 파일 https://openslr.trmal.net/resources/17/musan.tar.gz 의 앞부분에서 FMA 실제 음악 10개를 확보했다. 원본 전체 다운로드를 피하기 위해 압축 전송량 상한을 두고 순차 접근했다. HF 미러의 music/default rows API는 모두 HTTP 500으로 실패했고, 공식 배포본으로 전환한 뒤 실제 음원 다운로드를 완료했다.

MUSAN 자체의 설명은 https://www.openslr.org/17/ 와 원 논문 https://arxiv.org/html/1510.08484 (2015), 개별 이용 조건과 연주자는 확보한 `LICENSE`, `ANNOTATIONS`에 따른다. 메타데이터 미러 리비전과 SHA는 `docs/14_musan-metadata-audit-20260912.md` 및 원 manifest에 있다. 공식 음원과 미러 메타데이터의 ID를 대응했으며 공식 전체 압축 체크섬을 확인한 것은 아니다.

실제 음악 표본은 Airglow 5곡, Alecs_Band 4곡, Arne_Huseby 1곡이다. 독립 연주자 10명이 아니므로 일반화 성능을 평가하는 충분한 표본이 아니다. 라이선스 원문과 곡명/연주자 및 출처 URL을 ZIP에 보존했다. 버전이 생략된 Attribution 표기를 임의의 CC 버전으로 바꾸지 않았다.

FakeMusicCaps 공식 원본 https://zenodo.org/records/15063698 에서 MusicGen_medium, audioldm2, musicldm, mustango, stable_audio_open의 출력 각 2개를 확보했다. 라이선스는 CC BY-NC 4.0이다. 생성기별 고정 시드로 선택하고 설명 ID가 서로 중복되지 않도록 했다. 이전에 확보한 같은 설명 ID 5개 표본을 성능 시험에 재사용하지 않았다.

확보한 원본 수·그룹 수는 `data/raw/music-pilot-20260912/manifest.json` 로컬 실측이다. 원본 총 용량은 92,000,474 bytes이며 원본과 크롭 각각 SHA-256을 기록했다. 생성 ZIP은 개별 항목 CRC를 검증했으나 전체 ZIP MD5는 검증하지 않았다.

진짜/생성 모두 중앙 8초, 16kHz, mono PCM16으로 맞췄다. 가짜만 길게 반복하지 않고, 가짜만 다른 포맷으로 저장하지 않는다. 비유한값·길이 부족·무음 크롭·범위를 벗어난 진폭은 거부한다. 원본 float32 WAV에서 비오디오 청크를 건너뛴다는 SciPy 경고가 있었고 오디오 디코딩과 크롭 검사는 통과했다.

## 이 실험에서 판단할 수 있는 것

현재 개발 코드의 direct/max, direct/mean, baseline fusion/max를 같은 파일에서 비교한다. direct/mean은 FILE만이 아니라 성분 세그먼트 집계도 함께 바꾼다는 것을 명시했다. 세 설정은 결과 확인 전에 정했다. 모델 학습은 하지 않는다.

파일의 생성 여부는 공개 데이터 출처에서 정답을 가져온다. 성분 단위 존재 여부·가창·부분 위조 라벨을 직접 청취해 검토한 자료가 아니므로 **FILE EER와 FILE ROC-AUC만** 산출한다. 대회 총점·Music EER·Voice EER·CPS·2차 환산점수를 계산하지 않는다. 2차 성능점수는 사용자가 제공한 식 `30 × (Public / 대상팀 최고 Public)^N`이 적용되며 N은 비공개이므로 이 작은 실험의 수치로 환산하지 않는다.

이 결과는 출처 편향이 있는 소규모 진단이다. 새 모델 채택·대회 점수 향상·미지 생성기 일반화를 확정하지 않는다. 이후 독립 검증 데이터 구성 시 이번에 본 원곡·설명 그룹은 최종 확인셋에서 제외해야 한다. 과거 FakeMusicCaps 분할 계획 CSV는 실제 데이터 반입 전 계획이므로 이 노출 목록과 함께 검토해야 한다.

## 실행 검증 범위

로컬 테스트 3개 통과:

1. ZIP CRC, 크롭 20개의 저장 해시, mono/16kHz/8초/PCM16 검사.
2. 생성 노트북의 코드 셀 문법, 세 설정 변경, 순서를 뒤집은 가짜 예측의 ID 결합·EER/AUC 계산 검사.
3. 잘못된 ID 거부, 오류 기록 저장, 실패 후 원본 스크립트 복원 검사.

테스트에 사용한 가짜 예측의 완벽한 점수는 도구 검사용이며 모델 성능이 아니다. 실제 신규 GPU 실험은 아직 실행하지 않았다. 모델 설치·다운로드 경로는 사용자 첫 Colab 실행에서 성공한 코드를 재사용했다. 이전 `/content/submit/model` 가중치가 남아 있고 해시가 맞으면 새 실험 폴더에서 재사용한다.

모델 가중치 확인은 실패하면 중단한다. 예측값의 행/ID/컬럼/범위/유한값을 검사하고 폴백 발생을 거부한다. 설정별 프로세스 전체 시간과 로그를 저장하며 대회 전체 시간으로 부풀려 환산하지 않는다. 추론 프로세스에는 Hugging Face/Transformers 오프라인 설정을 적용한다. 이는 모든 OS 네트워크를 차단한 제출 서버 검증과 동일하지 않다.

## 실행과 결과 전달

1. Colab에서 새 `colab_music_pilot.ipynb`를 업로드한다.
2. 무료 GPU를 선택한다.
3. 위에서부터 실행하고 파일 업로드 창에서 `music_pilot_bundle.zip`을 선택한다. 예전 `verify_bundle.zip`을 넣으면 해시 검사에서 중단한다.
4. 마지막 비교 셀이 끝나면 `music_pilot_results.zip`이 다운로드된다. 다운로드 허용 안내가 나오면 허용한다.
5. 결과 ZIP을 다운로드 폴더에 두고 Codex에게 알려준다. 오류가 발생하면 출력이 포함된 ipynb도 저장한다.

실험 파일을 DACON에 올리지 않는다. 최종 대회 제출물은 별도의 `submit.zip`이고 루트는 `model/`, `script.py`, `requirements.txt`만 둔다. 평가 서버의 읽기 전용 `data/`를 수정하지 않으며 `output/submission.csv`에 쓴다. 사용자 제공 안내의 마지막 `open/` 표기는 앞부분 구조와 불일치하지만 현재 배포·실행 경로는 `data/`다. 모델 다운로드는 Colab 준비 단계이며 제출 추론 중에는 허용하지 않는다. 기본 설치 패키지 재설치, 10GB/32GB, 설치 10분/추론 60분 조건은 최종 제출 전 별도 검사한다.

AI 활용 기록: 2026-09-12, Codex가 실제 음악·생성 음악 소량 확보, 출처/해시 기록, 동일 길이·포맷 크롭, 진단 노트북·번들 제작, 모의 예측을 이용한 로컬 오류 검사를 수행했다. 실제 GPU 성능 측정·학습·대회 제출은 수행하지 않았다.
