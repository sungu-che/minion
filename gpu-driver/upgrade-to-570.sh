#!/usr/bin/env bash
# Upgrade a Brev L40S box from the image's held NVIDIA driver 565.57.01 to 570.211.01 (what trida-stack and
# GravityOCR were validated on; CUDA 12.8). Run as a sudo-capable user. Reboot afterwards, then run verify.sh.
#
#   ./upgrade-to-570.sh plan      record state, unhold 565, simulate; changes nothing but the holds
#   ./upgrade-to-570.sh apply     swap 565 -> 570 in one apt transaction, hold 570
#
# Why: the image holds every NVIDIA driver package <= 565, and toolkit 1.20.1 crashes generating CDI specs with
# driver 565 (libnvidia-sandboxutils: free(): invalid pointer), so no container can use the GPU.
set -euo pipefail
STATE=/var/tmp/driver-upgrade-$(date +%Y%m%d)
V=570.211.01-0ubuntu1                       # NVIDIA CUDA repo builds (not Ubuntu's -server packages, which pull 580)
# nvidia-smi ships in libnvidia-compute-570 here (NVIDIA's nvidia-utils-570 conflicts with it), so it is not listed.
# libnvidia-compute-570 needs nvidia-persistenced >= 570, so it is pinned to 570 (else apt takes 620.70.06); the
# unversioned libnvidia-cfg1 (620) is blocked so libnvidia-cfg1-570 provides it.
WANT=(cuda-drivers-570 nvidia-driver-570=$V nvidia-modprobe=$V nvidia-persistenced=$V
      cuda-drivers-fabricmanager-570 libnvidia-nscq-570=570.211.01-1)
# nvidia-prime (laptop GPU switching, held by the image) conflicts with nvidia-driver-570; it is the only installed
# package outside the 565 driver that is removed. libnvidia-cfg1 is not installed; "-" only keeps apt from adding it.
EXTRA_REMOVE=(nvidia-prime libnvidia-cfg1)
sudo mkdir -p "$STATE"

old_pkgs() { dpkg -l | awk '$1 ~ /^(ii|hi)$/ && $3 ~ /^565\.57\.01/ {print $2}' | sed 's/:amd64$//' | grep -vx nvidia-modprobe; }

case "${1:-}" in
  plan)
    # Keep the first run's records (a second plan run happens after the holds are already removed)
    [ -f "$STATE/holds-before.txt" ] || {
      dpkg -l | sudo tee "$STATE/dpkg-before.txt" >/dev/null
      apt-mark showhold | sudo tee "$STATE/holds-before.txt" >/dev/null
      nvidia-smi | sudo tee "$STATE/nvidia-smi-before.txt" >/dev/null || true; }
    mapfile -t OLD < <(old_pkgs)
    printf '%s\n' "${OLD[@]}" | sudo tee "$STATE/565-packages.txt" >/dev/null
    sudo apt-mark unhold "${OLD[@]}" nvidia-modprobe "${EXTRA_REMOVE[@]}" >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -s --no-install-recommends "${WANT[@]}" $(printf '%s- ' "${OLD[@]}" "${EXTRA_REMOVE[@]}") \
      | sudo tee "$STATE/simulation.txt" | grep -E '^(Inst|Remv)' | awk '{print $1, $2, $3, $4}'
    echo "--- driver packages installed at a version other than 570.211.01 (expect only build tools):"
    grep '^Inst' "$STATE/simulation.txt" | grep -i nvidia | grep -v '570\.211\.01' || echo "(none)"
    echo "state saved in $STATE"
    ;;
  apply)
    mapfile -t OLD < <(cat "$STATE/565-packages.txt")
    # Refuse if the transaction would remove anything that is not part of the 565 driver
    extra=$(grep '^Remv' "$STATE/simulation.txt" | awk '{print $2}' | sed 's/:amd64$//' \
      | grep -vxF -f "$STATE/565-packages.txt" | grep -vxF -e nvidia-prime || true)
    [ -z "$extra" ] || { echo "refusing: would also remove: $extra" >&2; exit 1; }
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${WANT[@]}" \
      $(printf '%s- ' "${OLD[@]}" "${EXTRA_REMOVE[@]}") 2>&1 | sudo tee "$STATE/apply.txt" | tail -5
    dpkg -l | awk '$1 == "ii" && $3 ~ /^570\.211\.01/ {print $2}' | sed 's/:amd64$//' \
      | xargs sudo apt-mark hold >/dev/null
    echo "570 installed and held. Reboot now: sudo reboot"
    ;;
  *) echo "usage: $0 plan|apply" >&2; exit 2 ;;
esac
