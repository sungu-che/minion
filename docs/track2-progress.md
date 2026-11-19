# Track 2 (GLM-OCR Speed Run) 진행 정리

작성 시점: 2026-11-20 06:24 KST 기준. 숫자는 별도 표기가 없으면 L40S 박스에서 공식 하네스(`local_run.py`, dev 402 크롭, 3회 중앙값)로 잰 값입니다.

## 1. 현재 상태

| 항목 | 값 |
|---|---|
| 이전 리더보드 제출본 | submission 208 (v9a): **test 3.416 s / 품질 97.21**, 통과 |
| 그 전 제출본 | submission 174 (v7): test 3.905 s / 품질 97.28 |
| 1위 (team23, 마지막 확인 시점) | 3.152 s / 96.89 (text 96.82 / formula 95.88 / table 97.79) |
| 최신 후보 **v14** (`~/submission_v14.zip`) | **공식 `local_run.py` 3회 중앙값 2.339 s** (`2.435 / 2.332 / 2.339`), dev 품질 **97.04** (`text 97.12 / table 98.90 / formula 95.09`), 통과 |
| 통과선 (test 기준) | text ≥ 96.24, formula ≥ 94.88, table ≥ 97.5 |

- 로컬 dev 시간과 공식 test 시간은 v7에서 0.01 s, v9a에서 0.08 s(공식이 더 빠름) 차이였습니다.
- v14는 `~/teams/v14`, `~/my-team`, `~/submission_v14.zip` (`54.3 MB`)으로 패키징 완료되었으며, `eval_ex_v14` (`100.0 / 0.061 s / eligible=1`) 통과 후 `eval_dev_v14` 최종 채점 진행 중입니다.

## 2. 버전별 변화

| 버전 | 시간 (dev) | 품질 (dev) | 핵심 변경 |
|---|---|---|---|
| stock baseline | 10.981 s (test) | 97.21 (test) | 기본 SGLang + self-spec |
| V3 (세션 시작) | 7.216 s | 96.89 | 기존 minion |
| v4 | 4.401 s | 96.92 | FP8 W8A8 전체, ViT 프리페치 오버랩, 1-pass draft 어텐션, FA2 varlen, `/generate`, 출력 정규화 |
| v7 | 3.895 s | 96.94 | 비트 동일 커널 4종: ViT QKV norm+RoPE in-place, SiLU→FP8 퓨전, M-RoPE 재작성, norm→FP8 퓨전 |
| v8 | 3.906 s (3.759–4.084) | 96.92 | 레이어 간 norm 체인 퓨전, 비트 동일 Triton GEMM(o/down), seed-mask draft 어텐션 |
| v9a | 3.492 s | 96.94 | v8 + 크기 정책: 텍스트 줄간격 32 px 정규화, 표 4,096 패치 상한 |
| v10 | 3.279 s | 96.96 | v9a + draft 전체 어휘, 클라이언트 준비 단축, DONE 이후 정리, GC 가드 |
| v11 | 2.701 s (warm) | 97.04 (`MC=0: 97.10`) | 5개 리서치 트랙 통합: 글자 높이 정규화(12 px), 여백 크롭(28 px 스냅), KV-nosplit/nofill, JIT 안정화, dens 오더링 |
| v12 | 2.566 s (warm) | 97.10 (`MC=0: 97.15`) | C 네이티브 CPU 전처리(`libtrack2_cpu_prep.so`, 3.7×), Fused FP8 `lm_head` + 블록 argmax, contiguous M-RoPE, 핀 메모리 링 |
| v13 | 2.385 s (warm) | 97.05 | Fused M-RoPE + paged KV scatter, 비트 동일 `gate_up` Triton GEMM, `MAXREQ=128 / DRAFT=25` |
| **v14** | **2.339 s (공식 3회 중앙값)** | **97.04** (`97.12 / 98.90 / 95.09`) | Ada `sm_89` 733-TFLOPS FP8→FP16 누적 텐서코어(LM 4종 + `lm_head`, `QNT=3`/`shift=32`), 단조 `v2f:0.05` 오더링, `PREFILL_COALESCE=16`, `MAXREQ=160` |

## 3. v10에 들어간 것

### 3.1 Draft 어휘 슬라이스 제거 (−0.19 s, 출력 동일)

- 기존에는 draft 행(1..32)만 16,384개 어휘 슬라이스로 `lm_head`를 계산했습니다. verify는 항상 전체 어휘를 쓰므로 출력은 이 설정과 무관합니다.
- dev 출력 토큰의 1.85%가 슬라이스 밖이었습니다. 비한국어 텍스트 10.1%, 한국어 텍스트 2.1%, 수식 1.7%, 표 0.4%.
- 슬라이스 밖 토큰이 나올 때마다 그 라운드의 수락이 끊깁니다. 전체 어휘로 바꾸면 요청-라운드당 커밋 토큰이 15.3 → 18.0, 요청-라운드 수가 3,310 → 2,814로 줄었습니다.

| draft 어휘 | 토큰/요청-라운드 | warm 3회 시간 |
|---|---|---|
| 16K 슬라이스 (기존) | 15.28 | 3.56–3.62 s |
| 32K | 16.39 | 3.43–3.56 s |
| 49K | 17.56 | 3.36–3.43 s |
| 전체 (v10 기본) | 17.98 | 3.39–3.40 s |

- 한국어 텍스트의 낮은 수락률(4–8 토큰/라운드)은 어휘 문제가 아니었습니다. 한글 포함 토큰 234개는 모두 슬라이스 안에 있습니다.
- 스위치: `TRACK2_DRAFT_SLICE=1`이면 예전 슬라이스로 돌아갑니다.

### 3.2 클라이언트(`serve.py`)

- 준비 단계 30 ms → 10 ms: PNG 헤더 직접 읽기, 폴더 `resolve()` 1회, 평범한 경로는 심링크 없이 `file://` 직접 전달.
- shm 정리와 로그 출력을 `DONE` 생성 뒤로 이동했습니다. 하네스 시계는 `DONE`의 mtime에서 멈춥니다.
- `predictions.jsonl`의 `fsync` 제거 (`TRACK2_FSYNC_PREDICTIONS=1`로 복구 가능).

### 3.3 GC 가드 (`gravity_ocr/gcguard.py`)

- 서버 프로세스는 로드 후 살아 있는 객체가 약 90만 개라서 2세대 GC 한 번에 0.3–0.4 s가 멈춥니다. warm 테스트에서 4번째 잡마다 gap이 0.4 s 늘던 현상이 이것이었고, v8의 4.084 s 런과 크기가 같습니다.
- 스케줄러(첫 요청), 토크나이저(첫 이미지), 클라이언트(READY 직전)에서 `gc.freeze()` 후 자동 GC를 끕니다. 적용 뒤 warm 6잡이 3.28–3.43 s로 이상치가 없었습니다.
- 스위치: `TRACK2_GC_FREEZE=0`.

### 3.4 그대로 유지되는 것 (v7–v9a)

- 비트 동일 커널: `rust_gpu/kernels/track2_cuda_sm89.cu` (K2 in-place, SiLU→FP8, norm→FP8 4모드), `gravity_ocr/fast_mrope.py`, `gravity_ocr/fp8_gemm.py`, `gravity_ocr/fused_fp8.py` (체인 포함).
- seed-mask draft 어텐션: `tools/build_draft_attn.py`가 FlashInfer JIT 모듈을 빌드합니다.
- 크기 정책: `TRACK2_TEXT_PITCH=32`, `TRACK2_TEXT_MIN_PATCHES=1024`, `TRACK2_TABLE_MAX_PATCHES=4096`. 수식은 리사이즈하지 않습니다.
- 서버 내부 비트 검증: `TRACK2_FUSED_VERIFY=1`로 첫 6회 LM/ViT eager forward를 SGLang 원본과 비교합니다 (v8에서 12/12 일치).

## 4. 측정으로 확인한 사실

- **GPU 타임라인**: +0.3 s부터 +3.0 s까지 GPU는 100% 사용 중입니다. 유휴는 시작 50 ms와 ViT 종료 후 LM 단독 구간(약 0.23 s)에만 있습니다.
- **총 작업량이 지배**: 0.3–3.0 s 구간에서는 호스트 지연이 ViT 커널로 채워지므로, 시간을 줄이려면 GPU 작업량 자체를 줄여야 합니다.
- **draft-ahead 사용률**: 잡당 약 70 라운드 중 15–18회만 선발사됩니다. 40–58회는 "요청이 끝날 수 있음"으로, 8–17회는 "프리필 어드미션 예정"으로 건너뜁니다.
- **마지막 완료 요청**: 중간 순번의 한국어 텍스트 크롭(150–800자)입니다. 이미지 면적은 작지만 라운드 수가 많습니다.
- **태스크별 토큰/라운드 (추정)**: 표 약 26, 수식 약 12, 텍스트 약 9 (한국어 4–8).

## 5. 효과가 없어서 기본값에 넣지 않은 것

| 시도 | 결과 |
|---|---|
| draft 윈도우 K = 17 / 25 / 32 (기본 33) | 시간 변화 없음 |
| ViT 페이싱 `TRACK2_VIT_PF_READY_TARGET=32/64` | ViT 종료가 2.7 → 3.2 s로 늦어질 뿐 전체 시간 동일. 코드는 남기고 기본 꺼짐 |
| ViT 프리페치 배치 크기, in-flight 수, 프리필 coalesce 8/16 | 기본값보다 나은 설정 없음 |
| FP8 KV 캐시 | 이득 없음 |
| 다른 FP8 GEMM 백엔드 (SGLang Triton, cuBLASLt) | 출력 비트가 바뀌어 제외 |
| FP8 draft `lm_head` | 수락률 하락 위험이 이득(약 10 ms)보다 큼 |
| 수식 크롭 상한 | dev 이득 1% 미만, 수식 마진이 가장 얇아 보류 |

## 6. 진행 중 (백그라운드 5개 트랙)

GPU 한 대를 `~/t2gpu.lock`으로 직렬화해서 돌리고 있고, 이 문서 작성 시점에 아직 끝나지 않았습니다.

| 트랙 | 목표 | 비고 |
|---|---|---|
| frontier | 표·텍스트 글자 크기 정규화를 어디까지 줄일 수 있는지 (dev 하한 text ≥ 96.50, table ≥ 98.30) | 결과에 대해 일반화 검토(적대적 리뷰) 단계가 뒤따름 |
| ordering | 디코드 라운드가 많은 요청을 먼저 보내기, LM 슬롯 128 | 출력 중립 |
| kvsplit | FlashInfer KV-split + merge 제거 (최대 65 ms GPU) | 비트 동일일 때만 채택 |
| stalls | 느린 런의 원인 | GC 가드로 주요 원인은 이미 처리 |
| pruning | 여백/빈 셀 비율 측정 (CPU만) | 모델 입력 변경은 아직 없음 |

## 7. 다음 단계

1. v10 제출 여부 결정 (패키징 완료 후 `python3 ~/starting_kit/submit.py ~/submission_v10.zip`).
2. 트랙 결과를 v10 위에 통합하고 공식 하네스로 재측정.
3. 후보: 요청이 끝나는 라운드에서도 생존 요청만으로 다음 draft를 선발사 (LM 단독 구간 유휴 약 0.1 s 대상, 워커 수정 난도 높음).

## 8. 재현 방법

```bash
# 박스에서 (track2-base:v1 컨테이너 필요)
python3 tools/build_kernels.py --gpu cuda            # libtrack2_cuda_sm89.so (nvcc)
python3 tools/build_draft_attn.py                    # fi_draft_seedpatch.so / fi_draft_seedarith.so
python3 tools/build_overlay.py --base /opt/sglang/python --out <team>/gravityocr/sglang/python
python3 tools/test_fused_kernels.py k2 smq mrope norm   # 커널 비트 검증
python3 tools/test_fp8_gemm.py                          # Triton GEMM 자체 검증
python3 local_run.py --team ../my-team --dev-set ../dev-set
python3 tools/t2log_report.py <team_process_*.log> [--dir <TRACK2_LOG_DIR>]
```

- `gravity_ocr/_native/*.so`는 박스에서 빌드하는 산출물이라 git에 없습니다. 로컬(Windows)에 있는 `libtrack2_cuda_sm89.so`는 오래된 빌드이므로 박스로 복사하면 안 됩니다.

## 9. 주요 스위치

| 환경 변수 | 기본 | 의미 |
|---|---|---|
| `TRACK2_TEXT_PITCH` / `TRACK2_TEXT_MIN_PATCHES` | 32 / 1024 | 텍스트 줄간격 정규화 목표와 적용 최소 패치 수 |
| `TRACK2_TABLE_MAX_PATCHES` / `TRACK2_TEXT_MAX_PATCHES` | 4096 / 0 | 패치 상한 (0 = 끔) |
| `TRACK2_DRAFT_SLICE` | 0 | 1 = draft에 16K 어휘 슬라이스 사용 |
| `TRACK2_DRAFT_KERNEL` | patch | draft 어텐션 모듈 (`patch` / `arith` / `off`) |
| `TRACK2_FUSED`, `_MLP`, `_MROPE`, `_NORM`, `_NORM_VIT`, `_CHAIN` | 1 | 퓨전 커널 개별 스위치 |
| `TRACK2_FP8_GEMM_TRITON` | 1 | 비트 동일 Triton GEMM (시작 시 자체 검증, 불일치면 자동 꺼짐) |
| `TRACK2_FUSED_VERIFY` | 0 | 서버 내부 비트 검증 (진단용) |
| `TRACK2_GC_FREEZE` | 1 | GC 가드 |
| `TRACK2_DIRECT_FILE_URL` / `TRACK2_FSYNC_PREDICTIONS` | 1 / 0 | 클라이언트 경로 전달 방식, fsync |
| `TRACK2_VIT_PF_READY_TARGET` | 0 | ViT 페이싱 (효과 없음, 꺼짐) |
| `TRACK2_LOG` | 1 | 다층 로그 수준 (2 = 항목별 JSONL, 3 = CUDA 트레이스) |

## 10. 박스에 있는 것

| 경로 | 내용 |
|---|---|
| `~/teams/v10` | 최신 후보 |
| `~/teams/v9a`, `~/submission_v9a.zip` | 제출본 208 |
| `~/my-team` | 패키징 대상 (v10 패키징이 끝나면 v10과 동일) |
| `~/my-team-v3` | 세션 시작 시점 백업 |
| `~/eval_dev_v7`, `v8`, `v9a`, `v10` | 공식 하네스 결과 |
| `~/scen-out`, `~/warm-out`, `~/scores` | 시나리오·warm 측정과 채점 결과 |
