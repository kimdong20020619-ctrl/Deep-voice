# 03. 라이선스 원장

> **채택하는 순간 한 줄 적는다.** 막판에 소급 정리하면 반드시 누락된다.
> 규칙: 외부 데이터·사전학습 모델은 "누구나 접근 가능한 공개 자원이며 최소 비영리 목적으로의
> 사용이 허용된 경우" 사용 가능하고, **2차 평가 시 모든 요소의 출처를 명시해야 한다.**

## 판정 기준

| 판정 | 조건 |
|---|---|
| ✅ 채택 | 공개 접근 가능 + 비영리 사용 명시 허용 + 라이선스 원문 확인 완료 |
| ⚠️ 보류 | 라이선스가 모호하거나 EULA 서명·신청 절차가 필요 |
| ❌ 기각 | 비공개, 상업 라이선스, 재배포 금지로 학습 사용이 불가 |

**⚠️ 보류 상태의 자원은 학습에 넣지 않는다.** 나중에 빼려면 전체 재학습이다.

---

## 사전학습 모델 — 현재 채택분

전부 대회 공식 배포물 `open.zip > baseline_submit.zip > model/` 에서 왔다.
**주최측이 베이스라인으로 배포한 모델이므로 대회 사용은 명시적으로 허용된 상태다.**

| # | 자원 | 용도 | 출처 | 라이선스 | 확인일 | 판정 |
|---|---|---|---|---|---|---|
| 1 | **DF-Arena 1B** (`DF_Arena_1B_V_1`) | 음성·음악 성분 FAKE 확률 | HF `Speech-Arena-2025/DF_Arena_1B_V_1`, rev `fb6ce85de12c2c5a509d89114adaf827dd75f49f` | `license: other` — 상위 코드 MIT + 원저작 기여분 **Non-Commercial**. 상업 이용은 별도 계약 | 2026-09-05 | ✅ |
| 2 | **HTDemucs** `955717e8-8726e21a` | 음성/반주 분리 | demucs v4 사전학습 체크포인트 (Meta / `facebook/demucs`) | ⬜ **원문 미확인** — demucs 본체는 MIT, 사전학습 가중치 조건 별도 확인 필요 | 2026-09-05 | ⚠️ |
| 3 | **PANNs Cnn14** `Cnn14_mAP=0.431.pth` | 음성·음악 존재 확률 | PANNs (Kong et al.), AudioSet 사전학습 | ⬜ **원문 미확인** — PANNs 코드는 Apache-2.0 로 알려져 있으나 배포본에 라이선스 파일 없음 | 2026-09-05 | ⚠️ |
| 4 | **wav2vec2-XLS-R-1B config** | DF-Arena 프론트엔드 구성 | `facebook/wav2vec2-xls-r-1b` (config만 동봉) | ⬜ 미확인 (원본 모델은 Apache-2.0 로 알려짐) | 2026-09-05 | ⚠️ |

### 무결성 확인 (2026-09-05)

배포 `model/SHA256SUMS.txt` 와 로컬 전개본 SHA-256 이 **3개 전부 일치**한다.

```
780bc14fd4c15e65d58efdef728427cf03cd29cd60be528e97badf8c89087988  df_arena_1b/pytorch_model.bin
8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4  htdemucs/955717e8-8726e21a.th
0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31  panns/Cnn14_mAP=0.431.pth
```

### ⚠️ 재배포 금지

DF-Arena 1B 는 **비영리 라이선스**이고 나머지도 대회 배포물이다.
**공개 저장소에 올리지 않는다.** `.gitignore` 가 `submit/model/` 을 폴더째 제외한다.
제출 zip 에 동봉하는 것은 대회 규격이므로 문제없다.

### ⬜ 남은 확인 (2차 평가 전까지)

- [ ] HTDemucs 사전학습 가중치의 라이선스 원문
- [ ] PANNs Cnn14 체크포인트의 라이선스 원문
- [ ] `facebook/wav2vec2-xls-r-1b` 라이선스 원문
- [ ] DF-Arena LICENSE.txt Section 2·3 전문 정독 (대회 참가가 비영리 범위인지 최종 확인)

> 넷 다 주최측이 베이스라인으로 배포한 것이라 실무상 위험은 낮다.
> 다만 **출처 명시가 의무**이므로 보고서에 적을 원문 링크는 확보해야 한다.

---

## DF-Arena 1B 학습 데이터 — 데이터 설계에 직결

모델 카드(`model/df_arena_1b/README.md`)에 명시된 학습 데이터:

```
ASVspoof 2019, ASVspoof 2024, Codecfake, LibriSeVoc,
DFADD, CTRSVDD, SpoofCeleb, MLAAD, EnvSDD
```

**이 목록이 우리 데이터 전략을 규정한다.**

- 위 데이터셋으로 파인튜닝해도 **이득이 적다.** 이미 학습된 분포다.
- CTRSVDD(가창 음성)·EnvSDD(환경음)가 포함돼 있어 순수 음성 전용 모델이 아니다.
  다만 **악기·반주(instrumental) 생성 음악은 목록에 없다** → `MUSIC_FAKE_PROB` 이 도메인 밖인 근거.
- 따라서 차별화는 **목록에 없는 것**에서 나와야 한다:
  ① 한국어 TTS ② 전화채널·손실압축 ③ **생성 음악(instrumental)**

모델 카드 보고 EER (참고): `asvspoof_2019` 1.14% · `in_the_wild` 0.91% · `asvspoof_2021_la` 4.66%
· `codecfake` 8.37% · `add_2023_round_2` 11.54% · **`asvspoof_2024` 17.25%** · `add_2022_track_1` 22.21%

> 리더보드 1위 ADS 0.82323 을 세 EER 동일 가정으로 역산하면 평균 EER ≈ 17.7% 다.
> DF-Arena 의 어려운 벤치마크 성적대와 겹친다. **아직 zero-shot 근처에서 경쟁 중일 가능성이 있다.**

---

## 데이터셋 — 미채택 (P2에서 확정)

### 2026-09-11 원문 및 표본 확인

아래는 실제 학습 채택과 구분한 조사 상태다. 전체 구성과 분할 계획은 `13_music-data-audit-20260911.md`에 있다.

| 자원 | 확인된 이용 조건 | 공식 근거 | 상태 |
|---|---|---|---|
| FakeMusicCaps v2 | Zenodo API의 license.id = cc-by-nc-4.0; 저자 저장소 LICENSE도 일치 | https://zenodo.org/api/records/15063698 · https://github.com/polimi-ispl/FakeMusicCaps/blob/main/LICENSE | 공식 오디오 표본 확보. 보컬·성분 라벨 검토 전이므로 학습 미채택 |
| MUSAN | 배포 페이지 CC BY 4.0; 논문은 파일별 LICENSE·보컬 ANNOTATIONS 제공을 명시 | https://www.openslr.org/17/ · https://arxiv.org/html/1510.08484 | 진짜 반주 우선 후보. 원본·파일별 조건·주석 미확보 |
| SONICS | CC BY-NC 4.0 외 Suno/Udio 경쟁 제품 연구 제한 및 서비스 약관 준수 조건 명시 | https://github.com/awsaf49/sonics/blob/main/LICENSE | 보류. 보컬 포함·부분 생성 라벨도 성분별로 해석해야 함 |

DeepFense 미러의 Apache-2.0 표기를 FakeMusicCaps 원본 오디오의 이용 조건으로 사용하지 않는다. 원 배포자의 조건을 유지하고 출처·변경 내역을 기록한다.

| 자원 | 용도 | 라이선스 | 출처 URL | 확인일 | 판정 |
|---|---|---|---|---|---|
| _(P2에서 채운다)_ | | | | | |

### 후보 (전부 라이선스 미확인 ⬜)

- **Real speech**: LibriSpeech · Common Voice(ko) · AI Hub KsponSpeech
- **Fake speech**: 자체 생성 한국어 TTS (DF-Arena 미학습 영역) · MLAAD v5(이미 학습됨, 우선순위 낮음)
- **Real music**: FMA (Free Music Archive) · MUSAN music
- **Fake music**: FakeMusicCaps · SONICS ← **최우선 확보 대상**
- **Noise / RIR**: MUSAN noise · RIRS_NOISES
- 색인: https://github.com/media-sec-lab/Audio-Deepfake-Detection

> AI Hub KsponSpeech 는 신청·승인 절차가 있다. 국내 대회라 한국어 비중 확보가 중요하므로
> **P2 시작 시점에 바로 신청**해야 승인 대기로 일정이 밀리지 않는다.

---

## 자체 생성 자원

TTS 로 직접 생성한 딥보이스 데이터는 **사용한 TTS 모델의 라이선스**를 따른다.
생성물과 생성 도구를 둘 다 기록한다.

| 생성물 | 생성 도구 | 도구 라이선스 | 확인일 | 판정 |
|---|---|---|---|---|
| _(P2)_ | | | | |
