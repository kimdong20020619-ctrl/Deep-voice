# 44. 학습 헤드 추진 계획과 진행 상태 — 2026-09-24 ~ 09-29

**이 문서가 09-24 이후 작업의 단일 진행판이다.** 세션이 바뀌면 여기부터 읽는다.
사용자 결정(09-24): 대회는 Claude 가 주도, 논문은 사용자가 Codex 와 별도 진행. 하루 3회 제출을 모두 쓴다.

## 왜 학습인가

- 09-24 리더보드: 우리 305위 0.69949, 15위 0.87105, 1위 0.89871. 설정 A/B 는 +0.003~0.004 단위라 한계.
- 토크 게시판(운영진 답변): AI-Hub·CC BY-NC(-SA)·증강 데이터 허용, 참가자 질문이 전부 외부 데이터·증강·학습 관련
  → 상위권은 데이터를 합성해 학습한 것으로 추정(코드 비공개, 미확인).
- 우리 과거 프로브 실패 원인은 방법이 아니라 **표본 16~48개**였다. 이번엔 수천 개 규모로 간다.
- 운영진 09-16: 리더보드로 평가셋 구성을 역산하는 제출은 **불이익 가능** → 상수 진단 제출 금지.

## 데이터 v1 (라이선스는 확인 즉시 docs/03 에 기록)

| 역할 | 출처 | 규모 | 라이선스 | 상태 |
|---|---|---|---|---|
| 가짜 음성 (한국어) | HF `mueller91/MLAAD` `fake/ko/` — TTS 12종 × 999개 (Qwen3-TTS·Fish-S2-Pro·MiniMax·VoxCPM2·Chatterbox·OmniVoice·MOSS·Higgs·XTTS-v2·Bark) | 약 4.5 GB | CC BY-NC 4.0, **게이트(HF 로그인 + 약관 동의 필요)** | 목록 확인 |
| 진짜 음성 (한국어) | HF `Bingsu/zeroth-korean` (Zeroth, OpenSLR 40) | 10K~100K 발화 | CC BY 4.0, 비게이트 | 확인 |
| 가짜 음악 (반주) | Zenodo FakeMusicCaps v2 (DOI 10.5281/zenodo.15063698) — 부분 범위 다운로드 (`scripts/collect_fresh_fake_music.py` 방식) | zip 12.89 GB 중 일부 | CC BY-NC 4.0 | 기존 확인 |
| 진짜 음악 | Citizen DJ LOC-FMA 파일 단위 WAV (`scripts/collect_citizen_music.py`) + MUSAN music | 수백 곡 | 퍼블릭 도메인 / CC | 기존 확인 |

설계 원칙
- **생성기 단위 홀드아웃**: MLAAD 한국어 12종 중 2종, FakeMusicCaps 5종 중 1종은 학습에서 뺀다.
- 합성: 음성 단독 · 음악 단독 · 동시 혼합 · 순차 혼합. 라벨은 대회 정의대로(하나라도 가짜면 FILE=1).
- 증강: MP3 32~128 kbps, 전화채널(8 kHz μ-law + 300~3400 Hz 대역), 잡음, 음량. **REAL 에도 동일하게** 적용해 "후처리 = FAKE" 지름길을 막는다.
- 알려진 위험: 진짜=Zeroth 낭독 음색, 가짜=MLAAD 음색이라 **코퍼스 지름길**을 배울 수 있다. 진짜 음성 출처를 늘리고 증강으로 완화하되, 홀드아웃·리더보드로만 판단한다.

## 모델

DF-Arena 1B 고정 → fc5 입력 1280차원 세그먼트 임베딩 → 선형 헤드. 추론 경로는 이미 있다:
- `file_probe="probe"` + `model/file_head.npz` (FILE, blend 조절)
- `music_head="probe"` + `model/music_head.npz`
- `pipeline="direct"` + `model/head_{file,voice,music}.npz` (분리 없이 3헤드)

## 일정

| 일자 | 할 일 | 제출 |
|---|---|---|
| 09-25 | 데이터 노트북 작성·실행, 임베딩 추출 | `submit_topk25_directmax.zip` + 설정 후보 |
| 09-26 | 헤드 학습·홀드아웃 평가 → zip | 헤드 v1 변형 2~3개 |
| 09-27 | 결과 반영 v2 | 3개 |
| 09-28 | 최선 조합 확정, 최고점 선택 확인 | 여분 |
| 09-29 10:00 | 마감 — 새 실험 없음 | — |

## 진행 로그

- 09-24: 결과 topk25 0.69949(최고) · directmax 0.69882 · sonics 0.69007(폐기). 조합 zip 빌드 완료.
- 09-24: MLAAD 한국어 12종 확인, Zeroth·FakeMusicCaps 라이선스 확인. **대기: GPU 실행 계정·HF 게이트 동의(사용자)**.
- 09-24: 사용자 결정 — Kaggle GPU, 실행은 Claude 가 Chrome 으로, MLAAD 진행. Chrome 에서 Kaggle·HF 모두 로그아웃 상태 확인 → 로그인·전화인증·약관 동의·토큰 발급은 사용자 몫(보안 규칙).
- 09-24: 노트북 완성 — 원본 `scripts/kaggle_train_heads_src.py`, 생성기 `scripts/make_kaggle_train_notebook.py`,
  산출 `notebooks/kaggle_train_heads.ipynb` (셀 9, 169 KB, 제출 script.py SHA `d418dd6b…` 내장).
  - 규모: 학습 9,600 + 홀드아웃 1,400 클립. 홀드아웃 = MLAAD minimax·VoxCPM2, FMC stable_audio_open, Zeroth test 화자, MUSAN 해시 1/7.
  - 로컬 SMOKE=1 실행: 수집→합성→임베딩→학습→저장 전 구간 통과, 헤드 npz 가 `LinearProbe` 로 로드됨, 라벨 조합·증강 확인.
  - Kaggle 실행 조건: Accelerator GPU, Internet ON, Secrets `HF_TOKEN`(사용자 입력). 출력 `/kaggle/working/dv_heads_result.zip`.
  - 결과로 만들 제출 후보: (a) `file_probe=probe` blend 1.0 / 0.5 (b) `pipeline=direct` 3헤드 — 홀드아웃에서 DF-Arena 보다 좋은 헤드만.
- 09-24 밤 (무인 모드) 결정 로그:
  - MLAAD 없이 가능한 **음악 헤드부터** 학습. 근거: MUSIC 이 최약 축(EER 42.6%, 가중 0.27), SONICS(Suno 계열) 실패 → 평가 가짜 음악이 오픈 생성기 계열일 가능성(가설).
  - 음악 헤드는 추론과 같게 **혼합 클립은 HTDemucs 반주 스템 임베딩**으로 학습(제출의 music_head=probe 입력과 일치).
  - HF 토큰 없으면 MODE=music 자동 전환. Colab(Google 로그인 유지됨)으로 실행.
- 09-25: **첫 Colab 실행 소실** — 진행률 위젯 출력으로 탭이 멈추고 런타임이 회수됨. 09-25 제출도 0건(3슬롯 미사용).
- 09-26 재실행: 학습을 nohup 백그라운드 프로세스로 전환(`notebooks/colab_launch_heads.ipynb` ← `scripts/make_colab_launcher.py`,
  본체 `notebooks/train_heads_run.py`), 진행률 표시 끔, 셀은 3분마다 로그 꼬리만. 06:04 UTC 시작, DF-Arena·HTDemucs 해시/검증 통과.
- 결과 회수·빌드: 로그의 `DVHEAD` 줄 → `scripts/build_head_candidates.py <텍스트> musicprobe musicprobe05` (기준 topk25 + 변경 1개).
- 09-24 제출 정확값: topk25 ADS 0.6672777778 · directmax 0.6665396825 · sonics 0.6568095238.
