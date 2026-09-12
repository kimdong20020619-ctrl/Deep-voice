# MUSAN 음악 자료 대조 — 2026-09-12

## 현재 판단

MUSAN의 음악 주석은 확보했지만 이번에 조사한 FluidInference 미러에는 음악 음원이 없다. 이 미러를 음악 학습 데이터 다운로드 경로로 사용하면 안 된다. 현재 산출물은 원본 확보 후 대조할 목록이며, 학습·추론·점수 향상은 미검증이다.

공식 배포처는 https://www.openslr.org/17/ 이다. 공식 논문(2015) https://arxiv.org/html/1510.08484 의 2절은 개별 LICENSE의 출처·이용 조건과 ANNOTATIONS의 보컬·장르·연주자 정보를 설명한다. 공식 페이지의 전체 데이터셋 CC BY 4.0 표시만으로 개별 음원의 조건을 대체하지 않는다.

## 확보한 근거

- 도구: `scripts/inspect_musan_metadata.py` — Python 표준 라이브러리만 사용.
- 공개 미러: https://huggingface.co/datasets/FluidInference/musan
- 확인한 고정 리비전: `3edcfdf89b56dbe6a395ff29f9c29489e03d1321`
- 로컬 증거: `data/raw/musan-metadata-20260912/manifest.json`
- 검토 목록: 같은 폴더의 `music_inventory.csv`
- 메타데이터 원문은 파일별 URL·수신 바이트 수·SHA-256과 함께 저장했다. 공식 tar.gz와 미러의 바이트 동일성은 확인하지 않았다.

아래 수치는 해당 리비전의 저장된 목록과 원문을 로컬에서 집계한 결과다. MUSAN 공식 배포본 전체 수량으로 일반화하지 않는다.

| 항목 | 수치 | 출처 URL | 기준연도 | 비고 |
|---|---:|---|---|---|
| 미러 음악 음원 경로 | 0 | https://huggingface.co/api/datasets/FluidInference/musan | 2026 | 수신 목록은 manifest에 저장; 음악 경로의 WAV/FLAC/MP3/OGG 검사 |
| 확보 메타데이터 파일 | 12 | https://huggingface.co/datasets/FluidInference/musan/tree/3edcfdf89b56dbe6a395ff29f9c29489e03d1321/music | 2026 | README·ANNOTATIONS·LICENSE |
| 음악 주석 행 | 645 | 위 고정 리비전 음악 폴더 | 2026 | 고유 track ID, 다운로드 음원 수 아님 |
| 보컬 없음 N | 419 | 위 고정 리비전 ANNOTATIONS | 2026 | 주석 기준; 직접 청취 미실시 |
| 보컬 있음 Y | 226 | 위 고정 리비전 ANNOTATIONS | 2026 | 임의 크롭의 보컬 존재를 보장하지 않음 |
| LICENSE에서 ID가 검색되지 않은 행 | 1 | 위 고정 리비전 fma-western-art/LICENSE | 2026 | `music-fma-wa-0071`; 미허가 판정이 아니라 근거 확인 보류 |

## 점수 실험 전에 해결할 문제

1. **같은 연주자가 여러 폴더에 존재한다.** `John_Harrison`은 fma-western-art와 hd-classical에, `Kevin_MacLeod`는 fma-western-art와 rfm에 나온다. 출처 폴더만으로 분할하면 연주자 중복을 막을 수 없다. `Kevin_MacLoad` 등 유사 표기도 원문 대조 후 통합 여부를 판단해야 한다. 표기가 비슷하다는 이유만으로 자동 통합하지 않았다.
2. **같은 작품·녹음의 중복도 별도 검사한다.** 연주자 그룹을 분리해도 다른 표기나 재배포된 동일 녹음이 남을 수 있다. 원본 확보 후 해시와 오디오 중복 검사를 병행해야 한다.
3. **라이선스 근거에 불일치가 있다.** Jamendo LICENSE에는 `Attribution-ShareAlike 4.0 Unported (CC BY-SA 3.0)`처럼 버전이 서로 다른 표기가 있다. 이 항목을 자동 승인하지 않는다. ID가 문서에 있다는 사실은 이용 조건 검증 완료를 의미하지 않는다.
4. **곡 전체 보컬 주석과 크롭 라벨은 다르다.** Y인 곡의 전주를 잘라 VOICE_PRESENT=1로 넣으면 오라벨이 된다. 우선 N인 원본을 확보하고 실제 음악 존재를 확인한 뒤 짧은 음악 단독 실험을 준비한다.

CSV의 모든 행은 `eligible_for_training=false`, `review_status=metadata_only`, 분할 미정이다. 라이선스와 연주자 검토를 건너뛰고 자동으로 학습 목록에 편입하지 않는다.

## 실행 검증과 실패 기록

공개 API 조회는 샌드박스 네트워크 제한으로 처음 실패했고 승인 실행으로 완료했다. 웹 도구의 일부 raw URL 조회도 실패하여 공개 API와 고정 리비전 원문 다운로드를 사용했다.

초기 저장 코드에서 Windows 텍스트 줄바꿈 변환 때문에 수신 원문 해시와 로컬 파일 해시가 달라졌다. 저장 방식을 바이트 쓰기로 수정했다. 기존 파일은 줄바꿈 변환을 역변환한 후보가 **수신 당시 해시와 정확히 일치하는 경우에만** 복구했다. 해시 값을 새 파일에 맞춰 변경하지 않았다.

실제 확인 결과:

- 저장한 메타데이터 모두 수신 당시 SHA-256과 일치.
- 임시 폴더에서 재생성한 CSV가 기존 CSV와 바이트 단위로 일치.
- 메타데이터를 임의 변경하면 해시 불일치로 분석 중단.
- 주석 ID 중복 없음. 음원 학습이나 GPU 실행은 수행하지 않음.

재실행은 새로운 폴더명을 사용한다. 기존 출력은 덮어쓰지 않는다.

```powershell
Set-Location C:\Users\kimdo\Documents\GitHub\Deep-voice
python -X utf8 -B scripts/inspect_musan_metadata.py --out-dir data/raw/musan-metadata-new
```

## 다음 실행 단위

음원이 포함된 다른 배포 경로 또는 공식 아카이브에서 필요한 원본만 확보할 방법을 확인한다. 전체 아카이브를 받기 전에 저장 공간과 전송량을 확인한다. 원본과 주석·라이선스를 대조하고 연주자 그룹을 정리한 뒤 FakeMusicCaps와 길이·인코딩을 맞춘 파일럿을 구성한다. 그 후 모델 임베딩 추출과 음악 EER 비교가 가능하다. 현 단계에서 제출 파일을 바꾸거나 점수 상승을 예상 수치로 제시할 근거는 없다.

AI 활용 기록: 2026-09-12, Codex가 MUSAN 원문·미러 구조 확인, 제한된 메타데이터 다운로드, 주석/라이선스 목록 대조, 원문 해시 보존 오류 수정 및 재현 검증을 수행했다. 별도 대회 AI 활용 원장을 관리한다면 이 기록을 당일 행으로 옮긴다.
