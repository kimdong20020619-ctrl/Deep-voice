# 03. 라이선스 원장

> **채택하는 순간 한 줄 적는다.** 막판에 소급 정리하면 반드시 누락된다.
> 규칙: 외부 데이터·사전학습 모델은 "누구나 접근 가능한 공개 자원이며 최소 비영리 목적으로의 사용이
> 허용된 경우" 사용 가능하고, **2차 평가 시 모든 요소의 출처를 명시해야 한다.**

## 판정 기준

| 판정 | 조건 |
|---|---|
| ✅ 채택 | 공개 접근 가능 + 비영리 사용 명시 허용 + 라이선스 원문 확인 완료 |
| ⚠️ 보류 | 라이선스가 모호하거나 EULA 서명·신청 절차가 필요 |
| ❌ 기각 | 비공개, 상업 라이선스, 재배포 금지로 학습 사용이 불가 |

**⚠️ 보류 상태의 자원은 학습에 넣지 않는다.** 나중에 빼려면 전체 재학습이다.

## 데이터셋

| 자원 | 용도 | 라이선스 | 출처 URL | 확인일 | 판정 |
|---|---|---|---|---|---|
| _(미확인 — P2에서 채운다)_ | | | | | |

### 후보 (전부 라이선스 미확인 ⬜)

- **Real speech**: LibriSpeech · Common Voice(ko) · AI Hub KsponSpeech
- **Fake speech**: MLAAD v5 · ASVspoof 2019 LA · ASVspoof 2021 DF · In-the-Wild · WaveFake
- **Real music**: FMA (Free Music Archive) · MUSAN music
- **Fake music**: FakeMusicCaps · SONICS
- **Noise / RIR**: MUSAN noise · RIRS_NOISES
- 색인: https://github.com/media-sec-lab/Audio-Deepfake-Detection

> AI Hub KsponSpeech는 신청·승인 절차가 있다. 국내 대회라 한국어 비중 확보가 중요하므로
> **P2 시작 시점에 바로 신청**해야 승인 대기로 일정이 밀리지 않는다.

## 사전학습 모델

| 자원 | 용도 | 라이선스 | 출처 URL | 확인일 | 판정 |
|---|---|---|---|---|---|
| _(미확인 — P1에서 채운다)_ | | | | | |

### 후보 (전부 라이선스 미확인 ⬜)

- **Head A**: PANNs CNN14 · AST · BEATs (AudioSet 사전학습)
- **Head B**: `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` ·
  `garystafford/wav2vec2-deepfake-voice-detector` · `lab260/AASIST3` ·
  SSL-AASIST 저자 공개 가중치 (wav2vec2-XLSR-300M + AASIST, 약 318M)
- **Head C**: 미정

## 자체 생성 자원

TTS로 직접 생성한 딥보이스 데이터는 **사용한 TTS 모델의 라이선스**를 따른다.
생성물 자체와 생성에 쓴 모델을 둘 다 여기 기록한다.

| 생성물 | 생성 도구 | 도구 라이선스 | 확인일 | 판정 |
|---|---|---|---|---|
| _(P2)_ | | | | |
