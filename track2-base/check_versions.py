"""Record or check the pinned Track 2 toolchain.

  check_versions.py --write /etc/track2/pinned.json   at image build: record what is installed
  check_versions.py --check /etc/track2/pinned.json   at run time (harness): exit 1 if anything changed
"""
import json
import sys
from importlib import metadata

PACKAGES = ["torch", "torchvision", "triton", "flashinfer-python", "sglang-kernel", "sglang", "transformers",
            "nvidia-cudnn-cu12", "nvidia-nccl-cu12"]


def current():
    got = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for p in PACKAGES:
        try:
            got[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            got[p] = None
    try:
        import torch
        got["torch.version.cuda"] = torch.version.cuda
    except Exception:  # torch missing or broken: recorded as None, which the check will flag
        got["torch.version.cuda"] = None
    return got


def main():
    mode, path = sys.argv[1], sys.argv[2]
    got = current()
    if mode == "--write":
        json.dump(got, open(path, "w"), indent=2)
        print(json.dumps(got, indent=2))
        return
    want = json.load(open(path))
    bad = {k: {"pinned": v, "found": got.get(k)} for k, v in want.items() if got.get(k) != v}
    if bad:
        print(json.dumps({"changed": bad}, indent=2))
        sys.exit(1)
    print("pinned toolchain unchanged")


if __name__ == "__main__":
    main()
