import json
from pathlib import Path

import joblib
import sklearn

version = Path("artifacts/latest.txt").read_text().strip()
folder = Path("artifacts") / version

model = joblib.load(folder / "model.joblib")
meta = json.loads((folder / "metadata.json").read_text())

print("version:", version)
print("model:", meta["model"])
print("trained with sklearn:", meta["sklearn"], "| installed:", sklearn.__version__)
print("threshold:", meta["threshold"])
print("number of features:", len(meta["features"]))
print("OK, model loaded")