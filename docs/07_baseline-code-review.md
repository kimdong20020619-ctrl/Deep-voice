# 07. 베이스라인 `script.py` 코드 리뷰

> 대상: 대회 공식 베이스라인 제출물 `script.py` 원문 (2026-09-05 확보)
> `05_baseline-analysis.md`가 아키텍처 수준이라면, 이 문서는 **줄 단위 결함**을 다룬다.
> 실측하지 않은 것은 ⬜ 로 표시했다.

분류: **[완주]** 0점 위험 · **[점수]** EER 개선 · **[시간]** 60분 예산

---

## 🔴 [완주] 1. 예외 처리가 한 곳도 없다 — 단일 파일 실패가 전체를 죽인다

`main()` → `predict_presence_for_all_files` → `predict_fake_scores_for_all_files`
전 경로에 `try/except`가 없다. 다음 중 **하나라도** 발생하면 프로세스가 죽고 **1,200개 전부 0점**이다.

| 지점 | 실패 조건 |
|---|---|
| `load_audio` | `audio.size == 0` 또는 `np.isfinite` 실패 → `ValueError` |
| `librosa.load` | 손상 헤더·지원 안 되는 코덱 → 예외 |
| `load_track` (demucs) | ffmpeg 디코딩 실패 |
| `order_audio_files` | ID 불일치 → `ValueError` (아래 2번) |
| `apply_model` | OOM |

**평가 데이터 1,200개는 우리가 볼 수 없다.** 그중 하나가 0바이트이거나 헤더가 깨졌을 확률은 0이 아니다.
게다가 **런타임 오류는 일일 제출 3회 중 1회를 소진**한다 (설치 오류와 달리).

→ **파일 단위 `try/except` + 폴백 값**으로 감싼다. 실패 시 해당 행에 중립값을 넣고 계속 진행한다.
   1개 파일을 포기하는 손실 << 전체 0점.
   이 수정은 **점수와 무관하게 최우선**이다. 다른 모든 개선의 전제다.

## 🔴 [완주] 2. 확장자 화이트리스트가 좁고, 누락되면 즉시 크래시한다

```python
SUPPORTED_AUDIO_EXTENSIONS = {".aac",".flac",".m4a",".mp3",".ogg",".opus",".wav",".wma"}
```

`find_audio_files`가 이 집합에 없는 확장자를 **건너뛴다.** 그러면 `order_audio_files`에서
`missing_ids`가 발생해 `ValueError`로 죽는다.

대회 설명은 "MP3, WAV, FLAC **등** 다양한 오디오 확장자"다. **"등"이 위험하다.**
누락 후보: `.wave` · `.aif` / `.aiff` · `.au` · `.amr` · `.3gp` · `.mp4` · `.webm` · `.mka`
(전화채널이 포함된다고 했으므로 `.amr`은 특히 개연성이 있다)

→ 화이트리스트를 없애고 **`data/test/` 안의 모든 파일을 시도**한 뒤, 디코딩 실패한 것만 폴백 처리한다.
   ffmpeg가 시스템에 설치돼 있으므로 librosa/soundfile이 대부분 읽는다.

## 🟠 [점수] 3. `MUSIC_FAKE_PROB`을 음성 anti-spoofing 모델로 채운다

```python
music_fake = predict_fake(df_arena_model, ...)   # DF-Arena-1B = 음성 anti-spoofing
```

실효 가중치 **0.27** (음성 0.18보다 크다). 게다가 `FILE_FAKE`(0.45)가 여기 물려 있다.
→ `05_baseline-analysis.md` 4절 A항. **이 대회의 핵심 공략점.**

### 3-1. 연쇄 효과 — 음악 헤드를 바꾸면 융합도 같이 바꿔야 한다

현재 `VF`와 `MF`는 **같은 모델에서 나오므로 스케일이 동일**하다. 그래서 `max`가 그럭저럭 동작한다.
음악 헤드를 다른 모델로 교체하는 순간 **두 점수의 분포가 달라지고 `max` 융합이 한쪽으로 편향**된다.
→ 음악 헤드 교체와 융합 재설계는 **한 묶음**이다. 따로 실험하면 결과를 해석할 수 없다.

## 🟠 [점수] 4. `max(VP·VF, MP·MF)` 융합이 순위를 왜곡한다

EER은 순위 기반이다. 존재 확률을 곱하면 **명백히 FAKE인데 존재 확률이 애매한 파일이 아래로 눌린다.**
(MP=0.6, MF=0.95 → 0.57 이 MP=0.99, MF=0.60 → 0.594 보다 낮게 매겨진다)

**0.45짜리 단일 최대 항목**인데 구현 비용은 거의 0이다.

비교 후보:
- 하드 게이팅 후 `max(VF, MF)` — 존재 확률은 게이트로만 쓰고 점수에서 뺀다
- noisy-or: `1 − (1−VF)(1−MF)`
- 존재 확률에 감마 보정: `max(VP^γ·VF, MP^γ·MF)`, γ<1 로 눌림 완화
- 학습 융합 (소량 검증셋 필요)

## 🟠 [점수] 5. HTDemucs에 16 kHz 오디오를 44.1 kHz로 업샘플해 넣는다

```python
waveform = load_track(audio_path, model.audio_channels, model.samplerate)  # 44.1 kHz
```

평가 데이터는 **전부 16 kHz로 표준화**돼 있다. 이를 44.1 kHz로 업샘플하면 **8 kHz 이상 대역이 비어 있는**
신호가 된다. HTDemucs는 44.1 kHz 풀밴드 음악으로 학습됐으므로 완전한 도메인 외 입력이다.
전화채널(8 kHz 대역제한) 샘플은 더 심하다.

→ 분리 품질 저하 → 스템에 아티팩트 → **anti-spoofing 모델이 아티팩트를 spoof 단서로 오인.**
   `05_baseline-analysis.md`에서 지적한 "분리 아티팩트 오염"의 **근원이 여기다.**

대응: ① 게이팅으로 분리 자체를 줄인다 ② 분리 스템과 원본을 함께 점수화해 비교
③ 스템 아티팩트가 실린 데이터로 파인튜닝해 내성을 만든다 (P3)

## 🟡 [점수] 6. 짧은 파일이 tile 반복 패딩된다 — 경계 불연속

```python
SEGMENT_SAMPLES = 64_600          # 16 kHz 기준 4.0375초
if audio.size < SEGMENT_SAMPLES:
    repeat_count = SEGMENT_SAMPLES // audio.size + 1
    audio = np.tile(audio, repeat_count)
```

대회 최소 길이가 **4초 = 64,000 샘플**인데 세그먼트가 **64,600 샘플**이다.
→ **4.0초 ~ 4.04초 파일은 전부 `np.tile`로 반복 이어붙여진다.**

파형을 반복 접합하면 이음매에 **불연속(클릭)** 이 생긴다. 이 클릭은 광대역 임펄스이고,
anti-spoofing 모델이 학습한 적 없는 인공 아티팩트다. → 오탐 요인.

→ zero-pad / reflect-pad / 마지막 프레임 페이드 중 무엇이 나은지 비교한다. 구현 비용 거의 0.
   해당 파일이 소수라도 EER은 순위 기반이라 몇 개의 오탐이 순위를 흔든다.

## 🟡 [시간] 7. DF-Arena 추론에 배치가 없다

```python
for start in get_segment_starts(audio.size):
    segment_tensor = torch.from_numpy(segment).to(device)   # 세그먼트 1개씩
    logits = model(input_values=segment_tensor)["logits"]
```

1분 파일이면 15 세그먼트 × 2 스템 = **30회 순차 GPU 호출**. L4가 대부분 논다.
→ 세그먼트를 배치로 묶는다. VRAM 22.4 GiB에 여유가 크다.

## 🟡 [시간] 8. fp32로 추론한다

DF-Arena-1B(10억 파라미터)를 fp32로 돌린다. **L4는 bf16을 지원**한다.
→ `torch.autocast("cuda", dtype=torch.bfloat16)` 로 **2배 이상 속도 이득**, 판별 정확도 손실은 미미.
   확보한 시간은 앙상블·TTA에 재투자한다 (속도는 점수가 아니다).

## 🟡 [시간] 9. 오디오를 파일당 두 번 디코딩한다

- 1차 루프: `librosa.load(sr=16000)` → PANNs
- 2차 루프: `load_track(sr=44100)` → HTDemucs

**서로 다른 리샘플링으로 각 파일을 2회 디코딩**한다. 1,200개면 2,400회.
→ 단일 루프로 합친다. L4 22.4 GiB면 PANNs(0.3GB) + HTDemucs(0.3GB) + DF-Arena-1B(fp32 4GB)를
  **동시에 상주**시켜도 여유가 크다. 현재는 PANNs를 `del` 하고 다시 로드하는 구조다.

## 🟡 [시간] 10. `apply_model`에 CPU 모델을 넘긴다 ⬜ 미검증

```python
return model.cpu().eval()          # 모델은 CPU에 상주
...
apply_model(model, wav, device=device, ...)   # 호출할 때마다 device 지정
```

demucs 4.0.1의 `apply_model`이 호출마다 `model.to(device)`를 수행한다면
**1,200회 × 약 320 MB CPU→GPU 전송**이 발생한다. ⬜ 소스 확인 필요.
→ 모델을 한 번만 GPU로 올려두고 재사용한다.

## ⚪ 11. 사소한 것

- `separate_voice_and_music`의 역정규화 `sources * std + mean` — 4개 소스에 각각 `mean`을 더하므로
  합이 원신호와 어긋난다. 다만 오디오의 DC 평균은 ~0이라 실질 영향은 무시할 만하다.
- `load_audio(mono=True)` — 평가 데이터에 스테레오가 있는데 다운믹스한다.
  채널 간 차이가 생성 오디오 단서일 수 있으나 우선순위는 낮다.
- `find_audio_files`가 stem 정렬을 하지만 직후 `order_audio_files`가 재정렬한다. 무해.
- `parse_arguments([])` — CLI 인자를 무시한다. 평가 서버는 인자 없이 실행하므로 문제없지만
  로컬 테스트가 불편하다. 개선판에서는 인자를 살린다.

---

## 손대지 말 것

**존재 탐지(PANNs)는 그대로 둔다.**
`predictions[:, indices].max()` 는 세그먼트 축·라벨 축 양쪽에 max를 건다.
"어디든 있으면 존재"라는 정의와 정확히 맞고, **순차 혼합**(앞은 음악, 뒤는 음성)에도 옳게 동작한다.
실효 가중치 0.10에 1위가 이미 AUC 0.989다. **개선 여지가 최대 0.001. 투자 금지.**

---

## 개선 순서

| 단계 | 항목 | 근거 |
|---|---|---|
| **P1-a** | 1·2 (예외 처리·확장자) | 점수 이전에 **완주**. 다른 모든 개선의 전제 |
| **P1-b** | 4·6 (융합식·패딩) | 0.45 항목, 구현 비용 ~0 |
| **P1-c** | 7·8·9 (배치·bf16·단일 루프) | 시간 확보 → 나중에 정확도로 재투자 |
| **P2~P3** | 3·5 (음악 헤드·분리 도메인 갭) | 본 승부처. 데이터 구축이 선행 |

P1-a 부터 P1-c 까지는 **모델을 전혀 바꾸지 않는다.** 즉 zero-shot 성능은 그대로 두고
안정성·융합·속도만 손보는 단계다. 이 상태로 한 번 제출해 **베이스라인 점수를 재현**하는 것이
이후 모든 실험의 기준선이 된다.
