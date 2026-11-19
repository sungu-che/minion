# Track 2: Gravity OCR Speed Run · starter kit

Make GravityOCR as fast as possible while keeping quality on the private test set above three lines:
text ≥ 96.24, formula ≥ 94.88, table ≥ 97.5 (0–100, in every timed run): the baseline's test scores minus 1 point.
GravityOCR 추론 속도를 최대한 끌어올리는 대회입니다. 모든 측정 런에서 비공개 테스트셋 점수가 텍스트 96.24,
수식 94.88, 표 97.5 이상이어야 순위에 들어갑니다(베이스라인 테스트 점수 − 1점).

## Get the kit onto your GPU box / 키트 다운로드

Everything below runs on the GPU box, in your home folder, with this layout: `~/starting_kit/` (this kit) and
`~/dev-set/` (the dev set). Your Submit server (제출 서버) account must be registered and approved for Track 2.
아래 커맨드는 모두 GPU 서버의 홈 디렉토리에서 실행합니다. 키트는 `~/starting_kit/`, dev 데이터는 `~/dev-set/`에 둡니다.

**A. From the command line (recommended):** on the box,
**A. 커맨드로 받기 (권장):** GPU 서버에서

```bash
curl -sO https://codabench-rw1y11331.gobrev.dev/board/get-kit.py && python3 get-kit.py   # writes ./starting_kit/ and ./dev-set/
```

**B. From the website:** download `starting_kit.zip` and `public_data.zip` from **Rules & files → Files** on your
laptop, copy both to the box (`brev copy ./starting_kit.zip <box>:~/`, the same for `public_data.zip`), then on the box:
**B. 웹사이트에서 받기:** **Rules & files → Files**에서 두 파일을 다운로드해 GPU 서버로 복사한 뒤, GPU 서버에서

```bash
sudo apt install -y unzip                       # once, if unzip is missing
unzip starting_kit.zip -d starting_kit          # both zips have no folder inside: -d makes it
unzip public_data.zip -d dev-set                # dev-set/{manifest.jsonl, targets.jsonl, images/}
```

## What's here / 구성

| Path | What it is |
|---|---|
| `team.yaml` | the one line our eval server runs: `serve: python3 serve.py` |
| `serve.py` | the reference team: plain GravityOCR, exactly what produced the baseline. Start here. |
| `local_run.py` | runs your submission on your GPU box exactly as our eval server does (same image, harness, scorer, restrictions) |
| `submit.py` | submits a zip from the command line (no browser needed; large zips with weights are fine); `--status` lists your submissions and scores |
| `programs/` | the harness and scorer `local_run.py` uses (the same code our eval server runs) |
| `track2-base/` | build files for the base image: the same Dockerfile and pins our eval server uses |
| `gpu-driver/` | upgrades a Brev L40S box from NVIDIA driver 565 to 570 (CUDA 12.8), which the base image needs |
| `examples/` | 3 dev crops with `manifest.jsonl`, `targets.jsonl` and a matching `predictions.jsonl` |

## Laptop and GPU box / 노트북과 GPU 서버

Use the website from your laptop (downloads, status, leaderboard) and the GPU box for building, running and big
files. Brev's CLI moves files between them; the box is also a normal SSH host after `brev refresh`.
웹사이트(다운로드, 제출 현황, 리더보드)는 노트북에서 보고, 빌드·실행·대용량 파일 작업은 GPU 서버에서 하세요.
파일 전송은 Brev CLI를 쓰면 됩니다.

| Direction / 방향 | Command / 커맨드 |
|---|---|
| Laptop → box (starter kit, dev set, code) | `brev copy ./starting_kit.zip <box>:~/` (then unzip: see above) |
| Box → laptop (small files only) | `brev copy <box>:~/my-team/serve.py ./` |
| Folders, resumable | `rsync -avP ./my-team/ <box>:~/my-team/` (or `scp -r`) |
| Code, both ways | a private git repo: `git push` on one side, `git pull` on the other |
| Edit on the box directly | `brev open <box> cursor` (or VS Code Remote-SSH) |

Changed weights are allowed (quantised, pruned, fine-tuned, ...): put them in your zip. Make and keep large files
on the box, and submit from there with `python3 submit.py`:
downloading GBs to a laptop only to upload them again is slow.
가중치 수정(양자화, 프루닝, 파인튜닝 등)도 허용되며, zip에 같이 넣어 제출하면 됩니다. 용량이 큰 파일은 GPU 서버에서
만들고 거기서 바로 `submit.py`로 제출하세요.

## 0. GPU box: NVIDIA driver 570 / GPU 서버: 드라이버 570

The base image needs driver 570 or newer (CUDA 12.8). Check with `nvidia-smi`. Brev's L40S image ships 565 (CUDA
12.7), held by apt; upgrade it once per box (about 5 minutes and a reboot):
베이스 이미지는 드라이버 570 이상(CUDA 12.8)이 필요합니다. Brev L40S 기본 이미지는 565라서 서버마다 한 번 업그레이드해야 합니다.

```bash
cd starting_kit
./gpu-driver/upgrade-to-570.sh plan     # records the current state, simulates; changes only apt holds
./gpu-driver/upgrade-to-570.sh apply    # 565 -> 570 in one apt transaction
sudo reboot
# after the reboot, from starting_kit/ again:
./gpu-driver/enable-docker-cdi.sh       # lets Docker hand the GPU to containers (local_run.py needs it)
./gpu-driver/verify.sh                  # driver, CUDA, a GPU container (run it again after step 1)
```

`enable-docker-cdi.sh` is needed on Brev's L40S image (Docker 27): it turns on Docker's CDI feature (backing up
`/etc/docker/daemon.json`), writes the GPU list `/etc/cdi/nvidia.yaml` if it is missing, restarts Docker and runs
`nvidia-smi` in a container. Run it once per box; it is safe to run again.
재부팅 후 `enable-docker-cdi.sh`를 한 번 실행하세요. Docker 컨테이너에서 GPU를 쓸 수 있게 설정해 줍니다.

## 1. Build the base image / 베이스 이미지 빌드

From `starting_kit/` (the dev set is next to it in `../dev-set/`):

```bash
docker build -t track2-base:v1 track2-base   # ~5 min on a Brev L40S box; downloads ~25 GB (CUDA, SGLang, weights, TeX Live)
./gpu-driver/verify.sh                       # now all three checks should say OK
```

`track2-base:v1` pins CUDA 12.8, PyTorch 2.9.1, Triton 3.5.1, FlashInfer 0.6.7.post2, SGLang 0.5.10 with the
GravityOCR self-speculative patch, and the GravityOCR weights ([`trillionlabs/GravityOCR-KO`](https://huggingface.co/trillionlabs/GravityOCR-KO)
at a fixed revision; public, no Hugging Face token needed). CUDA, PyTorch, Triton and FlashInfer cannot change, and
SGLang stays 0.5.10; your own weights, kernels and edited copy of trida-stack's `inference/gravityocr/` go in
your zip (see "Change the inference code and weights").
It also holds GravityOCR's evaluation code (trida-stack's `benchmark/gravityocr`), which scores your answers.

## 2. Run locally / 로컬 실행

```bash
python3 local_run.py --team . --dev-set ../dev-set        # from starting_kit/: runs the reference team
python3 local_run.py --team ../my-team --dev-set ../dev-set # your own folder, once you have one
```

Make your own folder next to the kit from the reference team: `mkdir ../my-team && cp team.yaml serve.py ../my-team/`.
레퍼런스 코드를 복사해서 팀 폴더를 만드세요: `mkdir ../my-team && cp team.yaml serve.py ../my-team/`

This runs 3 timed runs plus 1 untimed run, all on dev (locally dev stands in for the private test set), and prints
the same scores the leaderboard shows. It takes about 8 minutes for the reference team (most of it scoring). Your
folder must contain `team.yaml`; everything in it is your submission. Logs and predictions stay in the `outputs:`
folder it prints last (`output/team_process_*.log` is your code's own output).

**Locally the quality lines are checked on dev, not on the private test set**, so `eligible` there is only a guide.
The reference team scores about text 97.0, formula 95.4, table 98.1 on dev, and 97.24 / 95.88 / 98.5 on the
test set (the one that counts; the lines are those minus 1 point). Dev and test differ, so compare your dev scores
with the reference team's dev scores: a drop of more than about 1 point on any task is likely to fail on test.
로컬에서는 dev 셋으로 기준 점수를 체크하기 때문에 참고용입니다. 레퍼런스 코드는 dev에서 텍스트 97.0, 수식 95.4,
표 98.1, 비공개 테스트셋에서 97.24 / 95.88 / 98.5이며 기준은 테스트 점수 − 1점입니다. dev 점수는 레퍼런스 코드의
dev 점수와 비교하세요. 어느 항목이든 약 1점 넘게 떨어지면 테스트에서 통과하지 못할 가능성이 큽니다.

Quality is scored with GravityOCR's own evaluation code: text 1 − edit distance, tables TEDS (Markdown tables are
converted to HTML), formulas CDM (both LaTeX strings are rendered with pdflatex and the rendered characters matched;
answer formulas as LaTeX). Each task weighs 1/3. Scoring all four runs takes a few minutes (CDM is slow).

## 3. Submit / 제출

Zip the **contents** of your folder, then submit it from the command line (works on a GPU box with no browser, and
for large zips with weights) or upload it on the competition's **Submit** tab.
폴더 **안의 파일들**을 zip으로 묶어서 커맨드(`submit.py`)로 제출하거나 **Submit** 탭에서 업로드하세요. `submit.py`는
브라우저 없는 GPU 서버나 용량이 큰 zip에도 쓸 수 있습니다.

```bash
sudo apt install -y zip                                # once: Brev's image has no zip
cd ../my-team && zip -r ../submission.zip . && cd -   # from starting_kit/: zips everything inside my-team/
python3 submit.py ../submission.zip                    # first run: your Submit server username + password
```

`python3 submit.py --status` lists your recent submissions (status, quality, wall time, eligible).

Limits per team: **10 submissions per clock hour** (resets at :00), **at most 2 waiting or running** at once, and
**1 hour** per submission (all runs and scoring). A submission takes about 7–8 minutes for the baseline.
팀당 시간당 10회(매시 정각에 리셋), 대기·실행 중인 제출은 동시에 최대 2건, 제출 1건당 최대 1시간입니다.
`submit.py` logs in as your Submit server account (the same one as on the website), keeps the API token it gets back in
`~/.submit-server-token` (only you can read it; `--relogin` to sign in again) and prints the submission number. The
submission counts against your quota exactly like a browser upload.

**Keep your own copy of every zip.** To save disk, the site deletes a submission's zip about 2 hours after it
started unless it is your current leaderboard entry; its scores stay.
제출한 zip은 꼭 따로 보관하세요. 리더보드에 올라간 제출이 아니면 약 2시간 뒤 서버에서 zip이 삭제됩니다(점수는 남습니다).
`team.yaml` must be at the top level of the zip, not inside a subfolder.
`team.yaml`은 하위 폴더가 아니라 zip 최상위에 있어야 합니다.

```text
my-team/                      ← your folder
├── team.yaml                 ← required, at the top level
├── serve.py
├── kernels/                  ← anything your code needs: kernels, wheels, weights, configs
│   └── fused_draft.cu
└── weights/
    └── model-int8.safetensors
```

```text
submission.zip                 ✓ correct          submission.zip                 ✗ wrong
├── team.yaml                                     └── my-team/
├── serve.py                                          ├── team.yaml
├── kernels/…                                         └── serve.py
└── weights/…
```

## What's timed / 시간 측정 구간

One submission = 3 timed runs on the private test set + 1 untimed run on dev. Each run starts your code fresh.
제출 1건 = 비공개 테스트셋 3회(시간 측정) + dev 셋 1회(측정 안 함). 매 런마다 새 프로세스로 시작합니다.

```text
 your code starts         ready            test crops arrive                  answers in
 ───────●───────────────────●──────────────────────●═══════════════════════════════●──────── exit
        │ load model, warm up │      (waiting)      │  OCR all crops      ⏱ TIMED   │
        │   NOT timed         │                     │                               │
```

| | Timed? |
|---|---|
| Starting your code, loading the model, warm-up (up to 5 min) | ✗ |
| OCR of all crops, from the moment they are handed over until your answers are in | ✓ |
| Ranked wall time | median of the 3 test runs |
| Memory, files, background processes | cleared before every run |

## Start from `serve.py` / `serve.py`에서 시작

`serve.py` already handles the hand-off with the eval server. Keep it, and change the parts that do the work:

| In `serve.py` | What it does | Change it to go faster |
|---|---|---|
| `start_server()` | starts GravityOCR (patched SGLang, self-speculative decoding) | server flags; your own weights or `gravityocr/` code (below) |
| `ocr()` | one request per crop | request format, batching |
| `run_manifest()` | sends all crops, `CONCURRENCY` at a time | scheduling, ordering, concurrency |

### Change the inference code and weights / 추론 코드와 가중치 수정하기

GravityOCR's inference code is trida-stack's `inference/gravityocr/`. To change it, copy that folder into your team
folder as `gravityocr/`, edit it there, and `serve.py` runs your copy instead of the image's. The same for weights:

```text
my-team/
├── team.yaml
├── serve.py
├── weights/                          optional: your model folder (replaces /opt/gravityocr/weights)
└── gravityocr/                       optional: your copy of trida-stack's inference/gravityocr/
    ├── serve.sh                      the launcher (SGLang flags, speculative mode)
    └── sglang/
        ├── gravityocr_selfspec.patch GravityOCR's changes to SGLang 0.5.10
        └── python/sglang/srt/...     GravityOCR's own SGLang files (self-spec, ...)
```

Copy them out of the image (from `starting_kit/`, with your folder at `../my-team`):

```bash
mkdir -p ../my-team && docker run --rm -v "$(realpath ../my-team)":/out --entrypoint sh track2-base:v1 -c "cp -r /opt/trida-stack/inference/gravityocr /out/gravityocr && cp -r /opt/gravityocr/weights /out/weights && chown -R $(id -u):$(id -g) /out"
```

Copy only what you change: leave out the `weights` part to keep the stock weights (2.1 GB). Your `gravityocr/` can
also hold just the files you changed (same paths as in trida-stack): every file it lacks comes from the image.

What `serve.py` does with your copy, every run, before `READY` (untimed, a few seconds):

1. If you have `gravityocr/sglang/`: builds SGLang the way trida-stack's `sglang/setup.sh` does, with your files
   where you have them and the image's otherwise: SGLang 0.5.10 (from the image, no network) +
   `gravityocr_selfspec.patch` + the `python/` files (the image's, then yours on top).
2. Puts that SGLang first on `PYTHONPATH`, so the server and every process it starts import it.
3. Runs your `gravityocr/serve.sh` (or the image's), with `CKPT` set to your `weights/` if you have one.

So edit the files where they are in trida-stack:

- **A GravityOCR file** in `sglang/python/` (e.g. `speculative/selfspec_sampling.py`): edit it there.
- **A file the patch changes** (e.g. `models/glm_ocr.py`, `managers/scheduler.py`): either edit the patch, or put
  the whole edited file under `sglang/python/sglang/srt/...` (it is copied over the patched one). The patched version
  to start from is in the image at `/opt/sglang/python/sglang/srt/...` (see "Read the base image's code").
- **Launch flags**: `gravityocr/serve.sh`.

Check that your copy is used: `serve.py` logs `built SGLang: ... + patch <which patch> + python/ files ...`,
`starting <which serve.sh>`, `sglang=.../sglang-build/python` and `CKPT=<which weights>`. Run `local_run.py` and look in its
`team_process_*.log`; a `print(..., file=sys.stderr)` at the top of a file you changed shows up there too. A patch that
does not apply stops the server at load, with git's message in the log.

- `weights/` must be a **full model folder** like the stock one (`config.json`, `model.safetensors`, tokenizer and
  processor files, `chat_template.jinja`, `block_diffusion.json`), not just the safetensors.
- Only SGLang's **Python** code is rebuilt. The compiled parts (CUDA, PyTorch, Triton, FlashInfer, `sgl-kernel`) come
  from the image and cannot change; your own CUDA/Triton kernels go in your folder.

GravityOCR 추론 코드는 trida-stack의 `inference/gravityocr/`입니다. 이 폴더를 내 폴더에 `gravityocr/`로 복사해서
수정하면, `serve.py`가 이미지 대신 내 것을 사용합니다(가중치는 `weights/`).
- 위 커맨드로 이미지에서 복사하세요. 바꿀 것만 복사하면 됩니다. `gravityocr/`에 수정한 파일만 같은 경로로 넣어도
  되고, 없는 파일은 이미지 것을 사용합니다.
- 매 실행마다 `READY` 전에(시간 측정 안 함) 내 `gravityocr/sglang/`(패치 + `python/` 파일)로 SGLang을 다시 만들고,
  내 `serve.sh`로 서버를 띄웁니다.
- trida-stack에서 하던 그대로 수정하면 됩니다. 패치가 바꾸는 파일(`glm_ocr.py` 등)은 패치를 고치거나, 수정한 파일
  전체를 `sglang/python/sglang/srt/...`에 넣으세요(패치 결과 위에 덮어씁니다).
- `local_run.py`의 `team_process_*.log`에서 `built SGLang`, `starting ...`, `CKPT=...` 줄로 어떤 파일이 쓰였는지
  확인하세요.
  CUDA / PyTorch / Triton / FlashInfer 같은 컴파일된 부분은 바꿀 수 없습니다.

Writing your own `serve` program instead (still on the pinned SGLang)? The exact job protocol is on
**Rules & files → Submission format**.

### Where the model runs / 모델이 실행되는 위치

Everything runs in one container per submission, on the GPU, with no network:

```text
container from track2-base:v1
├── serve.py          your code (team.yaml starts it)
│     │  HTTP to http://127.0.0.1:30000
│     ▼
└── SGLang server     started by serve.py, model "glm-ocr", weights from CKPT
```

`start_server()` launches SGLang through the image's `serve.sh`; `ocr()` sends one crop per request. An HTTP server is
not required: only the job protocol (`READY`, `predictions.jsonl` + `DONE`, exit on `STOP`) is fixed.
제출마다 컨테이너 하나가 뜨고, 그 안에서 `serve.py`가 SGLang 서버(127.0.0.1:30000)를 띄워 크롭을 보냅니다.

The code `serve.py` calls lives in the base image:

| In the image | What it is |
|---|---|
| `/opt/trida-stack/inference/gravityocr/serve.sh` | launcher `start_server()` calls (SGLang flags) |
| `/opt/trida-stack/inference/gravityocr/ocr_request.py` | request format |
| `/opt/sglang/python/sglang/srt/models/glm_ocr.py` | the GravityOCR model in SGLang |
| `/opt/sglang/python/sglang/srt/speculative/` | self-speculative decoding |
| `/opt/trida-stack/inference/gravityocr/sglang/gravityocr_selfspec.patch` | our patch to SGLang |
| `/opt/gravityocr/weights/` | the stock weights |

### Read the base image's code / 베이스 이미지 코드 보기

Copy the source out once (38 MB) and open it in your editor, e.g. VS Code over SSH (`brev open <box> code`, or
Remote-SSH), then **File → Open Folder → `~/base-src`**:

```bash
mkdir -p ~/base-src && docker run --rm -v ~/base-src:/out --entrypoint sh track2-base:v1 -c "mkdir -p /out/sglang && cp -r /opt/sglang/python /out/sglang/ && cp -r /opt/trida-stack /out/ && chown -R $(id -u):$(id -g) /out"
```

A shell inside the image (deleted on `exit`):

```bash
docker run --rm -it --entrypoint bash track2-base:v1
```

Experiment with the GPU, your folder mounted at `/work` (edits there are kept):

```bash
docker run --rm -it --device nvidia.com/gpu=all -v ~/my-team:/work -w /work --entrypoint bash track2-base:v1
```

**Edits inside the image never count.** Every run starts from our unchanged `track2-base:v1`, and your code cannot
write to `/opt`. To change something, copy it into your folder (it goes in your zip): see "Change the inference
code and weights" above. Then check with `local_run.py`.
`~/base-src`는 읽기용입니다. 이미지 안의 파일을 고쳐도 채점에는 반영되지 않습니다. 바꾸려면 내 폴더(zip)에 넣고 거기서
불러오도록 `serve.py`를 수정한 뒤 `local_run.py`로 확인하세요.
