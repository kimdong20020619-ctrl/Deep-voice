# 음악 데이터 원문·표본 조사와 구축 계획

작성일: 2026-09-11. 내부 연구·개발용 점검 기록이며 공모전 제출 보고서 본문이 아니다.

## 판단

첫 음악 실험은 FakeMusicCaps 생성 음악과 MUSAN의 보컬 없는 실제 음악을 우선 검토한다. 이는 점수 개선이 검증된 조합이 아니라, 원본 출처와 반주 선별 근거를 추적할 수 있는 출발점이다. SONICS는 추가 이용 조건과 노래 단위 라벨 문제를 해결하기 전까지 보류한다.

현재는 가짜 음악 표본과 분할 계획만 확보했다. 진짜 음악 원본, 파일별 출처·주석, 충분한 검토 라벨, GPU 예측은 아직 없다. 학습셋 구축·성능 개선이 완료된 상태가 아니다.

## 원문 비교

| 항목 | 수치 | 출처 URL | 기준연도 | 비고 |
|---|---|---|---|---|
| FakeMusicCaps v2 | 27,605개, 생성기 5종, ZIP 12,889,873,014 bytes | https://zenodo.org/records/15063698 | 2025 | 공식 API와 ZIP 목록 대조. 가짜 오디오만으로 진짜/가짜 판별 평가를 완성할 수 없음 |
| MUSAN | 배포 페이지 압축 크기 표기 11G | https://www.openslr.org/17/ | 2015 | 보컬 없는 음악을 주석으로 선별할 근거가 있음. 파일별 라이선스·출처를 함께 보존해야 함 |
| SONICS | HF 저장소 표기 32.6GB | https://huggingface.co/datasets/awsaf49/sonics/tree/main | 2026 조회 | 가짜 오디오와 진짜 메타데이터 제공. 진짜 오디오 확보는 별도 작업 |

FakeMusicCaps 라이선스는 공식 API `https://zenodo.org/api/records/15063698`에서 `cc-by-nc-4.0`으로 확인했다. 저자 저장소 LICENSE도 동일하다: https://raw.githubusercontent.com/polimi-ispl/FakeMusicCaps/main/LICENSE

SONICS LICENSE는 CC BY-NC 4.0뿐 아니라 Suno/Udio의 경쟁 제품 연구 제한과 서비스 이용약관 준수를 명시한다. 이 추가 조건을 생략한 이전의 간략한 라이선스 설명만으로 채택하지 않는다: https://raw.githubusercontent.com/awsaf49/sonics/main/LICENSE

MUSAN은 파일별 LICENSE와 보컬 ANNOTATIONS를 제공한다고 원 논문에 명시한다. 오래된 데이터이므로 최신 생성기 분포를 재현하는 가짜 음악의 대체가 아니라 실제 원천 음악 후보로 쓴다: https://arxiv.org/html/1510.08484 (2015)

## 실제 ZIP 확인 결과

도구: `scripts/inspect_fakemusiccaps.py`. 전체 ZIP을 받지 않고 HTTP Range로 목록 및 표본을 읽는다. 서버가 범위를 무시하면 전체 다운로드를 거부하고, 실행별 전송 한도를 둔다. ZIP 전체 MD5는 검증하지 않았다. 읽은 개별 항목은 zipfile의 CRC 검사와 저장한 SHA-256으로 추적한다.

- 실제 오디오: 생성기별 5,521개, 총 27,605개.
- 추가로 `__MACOSX/.../._*.wav` 메타데이터가 같은 수만큼 있다. 확장자만 세면 55,210개로 잘못 집계된다.
- 조사 스크립트의 첫 실행에도 이 항목이 표본으로 섞였다. 필터를 고치고 실제 RIFF/WAVE 헤더까지 확인했다. 이전 조사 폴더는 원시 조사 기록이며 학습에 사용하면 안 된다.
- 합성 코드에도 macOS 메타데이터가 원본으로 들어오면 명시적으로 중단하는 검사를 추가했다.
- 서버가 부분 응답을 짧게 반환해 표본 재조사가 한 번 실패했다. 범위·길이 검사와 한도 내 재시도를 적용한 뒤 완료했다.

최종 원본 목록·표본:

`data/raw/fakemusiccaps-audit-20260911-v2/manifest.json`

| 생성기 | 표본 실제 길이(초) | 샘플레이트 | 형식 | 확인 범위 |
|---|---:|---:|---|---|
| MusicGen_medium | 10.18 | 16000 | mono float32 WAV | 디코딩·유한값·비무음 |
| audioldm2 | 10.0 | 16000 | mono float32 WAV | 동일 |
| musicldm | 10.0 | 16000 | mono float32 WAV | 동일 |
| mustango | 10.242 | 16000 | mono float32 WAV | 동일 |
| stable_audio_open | 10.0 | 16000 | mono float32 WAV | 동일 |

위 수치는 로컬 표본 실측이며 전체 분포가 아니다. 모든 표본은 같은 파일 stem의 생성기별 변형이다. 선정 방식은 구조 점검용이지 무작위 성능 평가가 아니다. SciPy WAV 디코더가 알 수 없는 비오디오 청크를 건너뛴다는 경고는 있었으며 오디오 배열은 정상으로 읽혔다.

청취·보컬 존재 판정은 미실시다. `voice_present`, `music_present`는 unreviewed이며 `eligible_for_training=false`로 저장했다. 음원 파일 확장자나 생성기명만으로 무보컬 반주라고 확정하지 않는다.

## 검증을 왜곡할 수 있는 지점

1. **같은 설명의 생성기별 변형 중복**: FakeMusicCaps는 MusicCaps 설명을 여러 생성기에 넣는다. 설명 ID가 같은 결과물을 다른 분할에 넣으면 생성기를 분리해도 내용이 중복된다. 원 논문: https://arxiv.org/html/2409.10684v2 (2024; 배포 v2는 2025)
2. **보컬 오라벨**: MUSAN의 music 전체 또는 SONICS 노래 전체를 VOICE_PRESENT=0으로 놓으면 안 된다. FakeMusicCaps도 생성 결과의 성분을 검토해야 한다. 기존 PANNs의 판단만 정답으로 삼으면 평가 대상 모델의 오류를 정답에 복제한다.
3. **길이·반복 흔적**: 짧은 가짜 음악만 60초까지 반복하고 진짜는 긴 원본에서 자르면 반복 패턴이 정답과 연결된다. 첫 비교는 양쪽을 동일한 4~10초 조건으로 자르고 인위적인 반복을 배제한다. 긴 파일 일반화는 별도 원본 확보 후 검증한다.
4. **출처·코덱 지름길**: 오래된 실제 녹음과 최신 생성 파일의 차이를 생성 여부로 착각할 수 있다. 양쪽에 같은 인코딩·리샘플링 조건을 적용하고 여러 실제 음악 출처를 추가해야 한다.
5. **검증셋 대표성**: 이 데이터는 공개 생성기 중심이며 비공개 평가셋의 생성기·언어·코덱 분포는 모른다. 공개 검증 점수 상승을 DACON 상승폭으로 환산하지 않는다.

## 실제 생성한 분할 계획

`scripts/plan_fakemusiccaps_split.py`가 원본 목록에서 macOS 메타데이터를 제외하고 계획 CSV를 만들었다.

산출물: `data/raw/fakemusiccaps-source-plan-20260911.csv`

| 분할 | 후보 파일 수 | 설명 그룹 수 | 생성기 |
|---|---:|---:|---|
| train | 11724 | 3908 | MusicGen_medium, musicldm, mustango |
| development | 822 | 822 | audioldm2 |
| holdout | 791 | 791 | stable_audio_open |

이는 다운로드 완료 수나 라벨 승인 수가 아니다. 고정 해시로 설명 그룹을 70/15/15 구간에 배정하고, 생성기 배정과 일치하는 항목만 계획에 남겼다. 다른 조합은 쓰지 않는다. 모든 행은 검토 전·학습 불가 상태이며 로컬 오디오 경로와 해시는 비어 있다.

CSV에서 설명 그룹과 생성기가 분할 간 겹치지 않는지 실제 검사했다. 최종 확인셋은 후보 확정 전에는 점수를 보지 않는다. 이 설계는 보수적인 첫 실험안이며, 최종 확인 생성기 하나만으로 모든 미지 생성기에 일반화한다고 결론 낼 수 없다. 같은 분할 안의 음향적 중복까지 검사한 것은 아니다.

MUSAN은 곡·아티스트 단위로 먼저 분리하고 해당 분할 안에서 자른다. 혼합 오디오는 서로 같은 분할의 음성·음악 원본끼리만 만든다. 생성 음악의 설명 ID와 실제 음악의 원본 ID가 대응하면 같은 그룹 경계도 유지해야 한다.

## 다음 실행 조건

1. MUSAN 원본의 ANNOTATIONS와 LICENSE를 확보해 무보컬·사용 조건을 확인한다. 대용량 전체 파일을 급하게 내려받지 않는다. 조사 시 C드라이브 여유 공간은 22,989,590,528 bytes였다(로컬 DriveInfo 실측).
2. 필요한 FakeMusicCaps 원본을 계획에서 제한적으로 추출하고 보컬·음악 존재 여부를 검토한다. 현재 표본 5개는 학습량으로 충분하지 않다.
3. 첫 실험은 작은 음악 단독 세트에서 기준선과 음악 프로브를 비교한다. 이 단계는 Music EER 진단이며, 존재 탐지 클래스와 음성 축이 없는 음악 단독 세트에서 대회 총점을 만들지 않는다.
4. 독립 음성 원본을 갖춘 혼합 개발셋으로 FILE/MUSIC 개선과 VOICE 손실을 함께 본다. 전체 클래스가 있을 때만 공식 총점으로 후보를 선택한다.
5. 프로브 학습·독립 평가·실행 비용 검증 후 제출 여부를 정한다. 현재는 GPU 학습 및 점수 상승 검증을 수행하지 않았다.

## 재현

```powershell
python -X utf8 -B scripts/inspect_fakemusiccaps.py --out-dir data/raw/fakemusiccaps-audit-new --samples 5
python -X utf8 -B scripts/plan_fakemusiccaps_split.py --manifest data/raw/fakemusiccaps-audit-new/manifest.json --output data/raw/fakemusiccaps-plan-new.csv
python -X utf8 -B scripts/selftest_validation_guards.py
```

기존 산출물을 덮어쓰지 않으므로 새로운 출력 경로를 사용한다. 원격 ZIP 크기·버전이 바뀌면 URL과 크기를 공식 API로 다시 확인해야 한다. 범위 요청을 지원하지 않는 서버에서는 이 방식으로 다운로드할 수 없다.

AI 활용 기록: 2026-09-11, Codex가 공식 데이터·라이선스·논문 원문 조회, 제한된 원본 표본 확보, 메타데이터 오집계 발견·수정, 오디오 수치 검사, 분할 계획 생성을 수행함. 별도 공모전 AI 활용 원장에도 당일 기록한다.
