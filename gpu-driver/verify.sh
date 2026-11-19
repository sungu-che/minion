#!/usr/bin/env bash
# After the reboot (and again after building track2-base:v1): check driver, CUDA, Docker's GPU list and a GPU container.
set -uo pipefail
ok=1
say() { printf '%-28s %s\n' "$1" "$2"; }

drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
case "$drv" in
  57[0-9].*|5[89][0-9].*|[6-9][0-9][0-9].*) say "driver" "OK $drv ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" ;;
  *) say "driver" "NOT OK: '${drv:-none}' (need 570 or newer: ./gpu-driver/upgrade-to-570.sh, then reboot)"; ok=0 ;;
esac

if nvidia-ctk cdi list 2>/dev/null | grep -q "nvidia.com/gpu=all"; then
  say "Docker GPU list (CDI)" "OK (nvidia.com/gpu=all)"
else
  say "Docker GPU list (CDI)" "NOT OK: run ./gpu-driver/enable-docker-cdi.sh"; ok=0
fi

if ! docker image inspect track2-base:v1 >/dev/null 2>&1; then
  say "GPU container" "SKIPPED: track2-base:v1 not built yet (step 1); run this script again after it"
else
  out=$(docker run --rm --device nvidia.com/gpu=all --entrypoint python track2-base:v1 -c \
    "import torch; x=torch.randn(2048,2048,device='cuda'); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'ok', bool((x@x).isfinite().all()))" 2>&1 | tail -1)
  case "$out" in
    *"ok True"*) say "GPU container" "OK $out" ;;
    *) say "GPU container" "NOT OK: $out"; ok=0 ;;
  esac
fi

[ "$ok" = 1 ] && echo "All checks passed." || { echo "Some checks failed (see above)."; exit 1; }
