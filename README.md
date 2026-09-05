# Deep-voice

[DACON 236749 딥보이스 범죄 대응 AI 탐지 모델 경진대회](https://dacon.io/competitions/official/236749/overview/description) 참가 레포.
주최 행정안전부·NIA, 주관 국립과학수사연구원, 운영 데이콘.

**개인 참가 · 리더보드 마감 2026-09-29 · 2차 평가자료 2026-10-05**

---

## 과제

오디오 1개당 5개 확률을 예측한다. 학습 데이터는 제공되지 않으며 참가자가 직접 구성한다.

| 컬럼 | 의미 |
|---|---|
| `FILE_FAKE_PROB` | 파일 전체가 FAKE일 확률 |
| `VOICE_FAKE_PROB` | 음성 성분이 FAKE일 확률 |
| `MUSIC_FAKE_PROB` | 음악 성분이 FAKE일 확률 |
| `VOICE_PRESENT_PROB` | 음성이 존재할 확률 |
| `MUSIC_PRESENT_PROB` | 음악이 존재할 확률 |

평가 데이터 1,200개 · 4~60초 · 16kHz · MP3/WAV/FLAC 혼재 · 모노/스테레오 혼재 · 일부 전화채널.

## 점수 구조

```
Score = 0.9 × ADS + 0.1 × CPS
ADS   = 0.5×(1−File EER) + 0.2×(1−Voice EER) + 0.3×(1−Music EER)
CPS   = 0.5×Voice Presence AUC + 0.5×Music Presence AUC
```

**총점 실효 가중치 — FILE `0.45` · MUSIC `0.27` · VOICE `0.18` · 존재 2개 합계 `0.10`**

여기서 나오는 결론 셋:

1. **음악 위조도(0.27)가 음성(0.18)보다 무겁다.** 딥보이스 대회라는 이름과 반대다.
2. `FILE_FAKE`(0.45)가 음악 헤드에 물려 있어 **음악 헤드가 실질 최대 레버리지**다.
3. 존재 탐지는 실효 가중치 0.10에 이미 AUC 0.989, 연산 비중 1.2%다. **손대지 않는다.**

## 접근

성질이 다른 3개 문제로 분리한다. 한 헤드가 실패해도 나머지가 산다.

```
INPUT ─┬─ PANNs Cnn14 ─────────────> VOICE_PRESENT / MUSIC_PRESENT
       └─ (게이팅) HTDemucs ─┬─ vocals ───> DF-Arena 1B ─> VOICE_FAKE
                             └─ accomp. ──> DF-Arena 1B ─> MUSIC_FAKE
                                              융합 ─────> FILE_FAKE
```

가장 무거운 항목은 `FILE_FAKE`(0.45)인데 베이스라인은 이를 **합성**으로 구한다.
DF-Arena는 파일 단위 spoof 탐지기이므로 원본을 직접 넣는 경로(`file_head`)를 열어뒀다.

그다음 승부처가 **음악 위조도**다. DF-Arena 1B는 음성·가창·환경음으로 학습됐지만
**악기·반주 생성 음악은 학습 목록에 없다** → 이 컬럼이 도메인 밖이다.
그 다음이 **데이터 합성**(전화채널·손실압축·한국어 TTS)이다.

## 현재 상태

**P1 완료 — 첫 제출 대기.**

공식 베이스라인 대비 바꾼 것:

| 분류 | 변경 | 효과 |
|---|---|---|
| 완주 | 파일 단위 예외 처리 | 1개 파일 실패로 전체 0점이 되는 것을 막는다 |
| 완주 | 확장자 화이트리스트 제거 | 목록 밖 확장자로 ID 불일치 크래시가 나던 것 |
| 시간 | `demucs_gating` | 0.166 → 0.087 s/오디오초 (T4 실측) |
| 시간 | DF-Arena 배치 + bf16 | 단일 경로와 편차 0.000000 |
| 시간 | 단일 루프·모델 동시 상주 | 파일당 디코딩 1회 |
| 점수 | `file_head` — FILE(0.45)을 원본에서 직접 채점 | DF-Arena는 파일 단위 탐지기다. 게이팅 시 추가 비용 0 |
| 점수 | 융합식·집계·패딩 CONFIG화 | 기본값은 베이스라인과 동일. A/B용 |

측정값과 근거는 [`docs/04_experiment-log.md`](docs/04_experiment-log.md).

## 사용법

```bash
python src/eval/metric.py            # 대회 산식 자가검증 (리더보드 값 재현)
python scripts/selftest_logic.py     # GPU 없이 순수 로직 검증 (60여 항목)
python scripts/build_submit.py       # 규격 검증 + submit.zip 생성
python scripts/make_verify_bundle.py # Colab 검증용 번들 (250KB)
python scripts/make_verify_notebook.py  # notebooks/colab_verify.ipynb 재생성
```

GPU 검증은 [`notebooks/colab_verify.ipynb`](notebooks/colab_verify.ipynb)를 Colab에서 돌린다.
로컬(i3-8145U / RAM 8GB / GPU 없음)에서는 학습도 추론도 하지 않는다.

실험은 [`submit/script.py`](submit/script.py) 상단 `CONFIG`를 **한 번에 하나씩만** 바꾸고
`build_submit.py`를 다시 돌린다. 리더보드 제출은 하루 3회뿐이다.

## 구조

```
docs/       설계·규정·실측 기록 (2차 평가 보고서의 원본)
scripts/    검증·빌드 도구
submit/     제출 패키지 그대로 — model/ · script.py · requirements.txt
notebooks/  Colab 검증
src/eval/   대회 산식 구현 — 로컬 검증의 측정 도구
src/        데이터 합성·학습 코드 (P2~P3에서 채운다)
data/       배포 데이터와 합성 학습셋 (git 제외)
```

## 문서

| 파일 | 내용 |
|---|---|
| [`00_competition-spec.md`](docs/00_competition-spec.md) | 대회 규칙과 남은 미확인 항목 |
| [`01_data-design.md`](docs/01_data-design.md) | 학습 데이터 설계 → 「학습데이터 구성 보고서」 원본 |
| [`02_model-design.md`](docs/02_model-design.md) | 모델 설계 → 「모델 개발 보고서」 원본 |
| [`03_license-ledger.md`](docs/03_license-ledger.md) | 데이터·모델 출처와 라이선스 |
| [`04_experiment-log.md`](docs/04_experiment-log.md) | 실측값·제출 이력·A/B 계획 |
| [`05_baseline-analysis.md`](docs/05_baseline-analysis.md) | 베이스라인 아키텍처 해부와 공략점 |
| [`06_scoring-and-submission.md`](docs/06_scoring-and-submission.md) | 평가 산식·제출 규격·서버 환경 |
| [`07_baseline-code-review.md`](docs/07_baseline-code-review.md) | 베이스라인 `script.py` 줄 단위 리뷰 |
| [`08_compliance.md`](docs/08_compliance.md) | 대회 규정 준수 점검 |
| [`09_model-survey.md`](docs/09_model-survey.md) | 모델·지표 객관 조사와 채택 판정 |

## ⚠️ 이 레포에 넣지 않는 것

`.gitignore`가 막고 있다. 실수로 커밋하지 않도록 주의한다.

- **`submit/model/`** — DF-Arena 1B는 **비영리 라이선스**이고 대회 배포물이다. 공개 저장소 재배포는 위반이다.
- **`data/`** — 대회 배포 데이터. 재배포 금지.
- **`submit.zip` · `verify_bundle.zip`** — `scripts/`로 언제든 재생성된다.

## 규정

- 비공개 평가 데이터를 이용한 추가 학습·튜닝·Pseudo-Labeling 금지
- **다른 파일의 정보·예측값·통계를 이용한 예측 또는 보정 금지** —
  EER이 순위 기반이라 테스트셋 전체 정규화가 유혹적이지만 명시적 실격 사유다.
  `build_submit.py`가 관련 패턴을 발견하면 빌드를 실패시킨다.
- 2차 평가 시 사용한 모든 요소의 출처 명시 의무 → `03_license-ledger.md`

상세는 [`08_compliance.md`](docs/08_compliance.md).
