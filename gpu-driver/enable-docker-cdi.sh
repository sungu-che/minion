#!/usr/bin/env bash
# Docker < 28.2 accepts CDI device requests (what Codabench's GPU worker sends: nvidia.com/gpu=all) only with
# "features": {"cdi": true} in /etc/docker/daemon.json. Backs the file up, sets the flag, writes the GPU list
# /etc/cdi/nvidia.yaml if `nvidia-ctk cdi list` shows no devices (after the 570 upgrade it can be missing), restarts
# Docker and runs nvidia-smi in a container. Safe to run again.
set -euo pipefail
F=/etc/docker/daemon.json
sudo cp -a "$F" "$F.bak-$(date +%Y%m%d%H%M%S)"
sudo python3 - "$F" <<'EOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d.setdefault("features", {})["cdi"] = True
json.dump(d, open(p, "w"), indent=4)
print(json.dumps(d, indent=2))
EOF
if ! nvidia-ctk cdi list 2>/dev/null | grep -q "nvidia.com/gpu=all"; then
    echo "no CDI GPU list yet: writing /etc/cdi/nvidia.yaml"
    sudo mkdir -p /etc/cdi
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
fi
nvidia-ctk cdi list
sudo systemctl restart docker
docker run --rm --device nvidia.com/gpu=all ubuntu:22.04 nvidia-smi -L
