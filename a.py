"""
check_gpu_and_coverage.py
─────────────────────────
Run this to:
1. Check if PyTorch can see your GPU
2. Check full class coverage from your actual class_map.json

Run with:
    python check_gpu_and_coverage.py
    python check_gpu_and_coverage.py --class-map dataset_final/class_map.json
"""

import sys
import json
import argparse
from pathlib import Path

# ── GPU Check ────────────────────────────────────────────────────

def check_gpu():
    print("=" * 55)
    print("GPU / DEVICE CHECK")
    print("=" * 55)

    try:
        import torch
        print(f"  PyTorch version : {torch.__version__}")
        print(f"  CUDA available  : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version    : {torch.version.cuda}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU memory      : {mem:.1f} GB")
            print(f"\n  ✓ GPU is available and working!")
            print(f"  Training will be ~10-50x faster than CPU.")
        else:
            print(f"\n  ✗ No GPU detected by PyTorch.")
            print(f"\n  POSSIBLE REASONS:")
            print(f"  A) You have no NVIDIA GPU → CPU training is your only option")
            print(f"  B) You have an NVIDIA GPU but PyTorch was installed without CUDA")
            print(f"\n  TO CHECK: open Task Manager → Performance → GPU")
            print(f"  If you see an NVIDIA GPU there, run this to fix:")
            print()

            # Detect Python version for correct install command
            pv = sys.version_info
            print(f"  pip uninstall torch torchvision torchaudio -y")
            print(f"  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print()
            print(f"  (cu121 = CUDA 12.1, works for most modern NVIDIA cards)")
            print(f"  After reinstalling, run this script again to verify.")

        print()
        if torch.backends.mps.is_available():
            print("  Apple MPS (M1/M2) available too.")

    except ImportError:
        print("  ERROR: PyTorch not installed!")
        print("  Run: pip install torch")


# ── Coverage Check ───────────────────────────────────────────────

def check_coverage(class_map_path: str):
    print("=" * 55)
    print("CLASS COVERAGE CHECK")
    print("=" * 55)

    path = Path(class_map_path)
    if not path.exists():
        print(f"  ✗ class_map.json not found at: {class_map_path}")
        print(f"  Try: python check_gpu_and_coverage.py --class-map dataset_final/class_map.json")
        return

    with open(path, encoding="utf-8") as f:
        class_map = json.load(f)

    # class_map is {name: index}
    class_names = sorted(class_map.keys())
    print(f"  Found {len(class_names)} classes in {class_map_path}\n")

    # Import the mapping
    sys.path.insert(0, str(Path(__file__).parent / "transliteration"))
    try:
        from newa_to_devanagari import get_char_info
    except ImportError:
        print("  ✗ Cannot import newa_to_devanagari.py")
        print("  Make sure transliteration/newa_to_devanagari.py exists")
        return

    unmapped = []
    for name in class_names:
        info  = get_char_info(name)
        ok    = info["deva"] != "⟨?⟩"
        mark  = "✓" if ok else "✗ UNMAPPED"
        print(f"  {mark:12s} {name:25s} → {info['deva']}  ({info['iast']})")
        if not ok:
            unmapped.append(name)

    print(f"\n{'─'*55}")
    if unmapped:
        print(f"\n⚠  {len(unmapped)} UNMAPPED CLASSES:")
        for u in unmapped:
            print(f"     '{u}'")
        print(f"\n  Add these to CHAR_MAP in transliteration/newa_to_devanagari.py")
        print(f"\n  Template to paste in:")
        for u in unmapped:
            print(f'    "{u}":  {{"deva": "?",  "iast": "?",  "desc": "{u}"}},')
    else:
        print(f"\n  ✓ All {len(class_names)} classes are mapped!")


# ── Main ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--class-map", default="dataset_final/class_map.json",
                   help="Path to class_map.json (default: dataset_final/class_map.json)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    check_gpu()
    check_coverage(args.class_map)