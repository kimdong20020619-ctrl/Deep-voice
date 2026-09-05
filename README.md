# Deep-voice

[DACON 236749 딥보이스 범죄 대응 AI 탐지 모델 경진대회](https://dacon.io/competitions/official/236749/overview/description)
참가 레포. 주최 행정안전부·NIA, 주관 국립과학수사연구원, 운영 데이콘.

**개인 참가 · 리더보드 마감 2026-09-29 · 2차 평가자료 2026-10-05**

## 과제

오디오 1개당 5개 확률을 예측한다 — 전체 위조도 / 음성 위조도 / 음악 위조도 / 음성 존재 / 음악 존재.
학습 데이터는 제공되지 않으며 참가자가 직접 구성한다.

## 점수 구조

```
Score = 0.9 × ADS + 0.1 × CPS
ADS   = 0.5×(1−File EER) + 0.2×(1−Voice EER) + 0.3×(1−Music EER)
CPS   = 0.5×Voice Presence AUC + 0.5×Music Presence AUC
```

총점 실효 가중치 — **FILE 0.45 · MUSIC 0.27 · VOICE 0.18 · 존재 0.10**

## 접근

성질이 다른 3개 문제로 분리한다.

- **Head A** 음성·음악 존재 — AudioSet 사전학습(PANNs/AST/BEATs) zero-shot → 경량 파인튜닝
- **Head B** 음성 위조도 — SSL-AASIST(wav2vec2-XLSR + AASIST) 계열 파인튜닝
- **Head C** 음악 위조도 — 생성음악 탐지, 소수 데이터 + 강증강
- **결합** `fake_overall` — 평가 지표 확정 후 규칙 결합 / stacking 중 선택

승부처는 **음악 위조도**다. 실효 가중치가 음성(0.18)보다 큰 0.27인데,
공개 베이스라인은 이 컬럼을 음성 anti-spoofing 모델(DF-Arena-1B)로 채우고 있다.
`FILE_FAKE`(0.45)까지 이 헤드에 물려 있어 실효 레버리지는 그보다 크다.

그 다음이 **데이터 합성**이다. 평가셋 성질(4~60초 · 혼합 · 전화채널 · 다중 포맷 · 모노/스테레오)을
학습셋에서 재현하는 파이프라인이 `src/synth/`에 있다.

## 문서

| 파일 | 내용 |
|---|---|
| `docs/00_competition-spec.md` | 대회 규칙과 남은 미확인 항목 |
| `docs/05_baseline-analysis.md` | 공개 베이스라인 해부와 공략점 |
| `docs/06_scoring-and-submission.md` | 평가 산식·제출 규격·서버 환경 (확정) |
| `docs/01_data-design.md` | 학습 데이터 설계 → 2차 평가 「학습데이터 구성 보고서」 원본 |
| `docs/02_model-design.md` | 모델 설계 → 2차 평가 「모델 개발 보고서」 원본 |
| `docs/03_license-ledger.md` | 채택한 데이터·모델의 라이선스와 출처 |
| `docs/04_experiment-log.md` | 제출별 점수와 변경점 |

## 주의

이 레포는 **코드와 문서만** 담는다. `.gitignore`가 다음을 제외한다.

- `data/` — 대회 배포 데이터는 재배포 금지
- 모델 가중치(`*.pt`, `*.safetensors`, `submit/*.zip`) — 라이선스와 용량 문제

## 진행 상황

P0 완료(스펙 확정·베이스라인 해부). P1 착수 대기 — `open.zip` 필요.
남은 미확인 항목은 `docs/00_competition-spec.md` 참조.
