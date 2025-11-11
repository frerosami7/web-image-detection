import os
import sys
import json
from pathlib import Path

report = {
    "python": sys.version,
    "venv": str(Path(sys.executable).parent.parent),
    "cwd": str(Path.cwd()),
}

mods = {
    "numpy": "np",
    "torch": "torch",
    "torchvision": "torchvision",
    "anomalib.deploy": "anomalib_deploy",
    "imgaug": "imgaug",
    "open_clip_torch": "open_clip_torch",
    "cv2": "cv2",
    "streamlit": "streamlit",
}

for mod, alias in mods.items():
    try:
        m = __import__(mod, fromlist=["*"]) if "." in mod else __import__(mod)
        report[mod] = getattr(m, "__version__", "(no __version__)" )
    except Exception as e:
        report[mod] = f"ERROR: {e}"

# TorchInferencer check
try:
    from anomalib.deploy import TorchInferencer
    report["TorchInferencer"] = "import: OK"
except Exception as e:
    report["TorchInferencer"] = f"import ERROR: {e}"

# Optional: try to load bundled model if present
ckpt = Path("checkpoints/reverse_distillation_one_up.pt")
if ckpt.exists():
    try:
        from anomalib.deploy import TorchInferencer
        infer = TorchInferencer(path=str(ckpt), device="cpu")
        report["bundled_model_load"] = "OK"
    except Exception as e:
        report["bundled_model_load"] = f"ERROR: {e}"
else:
    report["bundled_model_load"] = "missing"

# Guidance flags
report["needs_torch"] = "yes" if str(report.get("torch", "")).startswith("ERROR") else "no"
report["needs_anomalib"] = "yes" if str(report.get("anomalib.deploy", "")).startswith("ERROR") else "no"

print(json.dumps(report, indent=2))
