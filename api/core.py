"""Artifact loading, input validation and scoring for the credit-risk API."""
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import sklearn

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))

# Hard limits reject nonsense. Values outside the training range only trigger a warning.
HARD_LIMITS = {
    "duration_months": (1, 120),
    "credit_amount": (1, 1_000_000),
    "installment_rate": (1, 4),
    "residence_since": (1, 4),
    "age": (18, 100),
    "existing_credits": (0, 10),
    "num_dependents": (0, 10),
}
WHOLE_NUMBER_FIELDS = {"duration_months", "installment_rate", "residence_since",
                       "age", "existing_credits", "num_dependents"}

NOTICE = (
    "Decision-support estimate only. The model was trained on 1,000 historical German "
    "credit records, its score is not a calibrated probability of default, and it must not "
    "be the sole basis for a lending decision. Age and other variables can act as proxies "
    "for protected characteristics. A qualified person should review every decision."
)


class ValidationFailed(Exception):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class ModelBundle:
    """One versioned artifact folder: model pipeline + metadata + schema."""

    def __init__(self, version: Optional[str] = None):
        if version is None:
            version = os.getenv("MODEL_VERSION") or (ARTIFACT_DIR / "latest.txt").read_text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", version):
            raise ValueError("Invalid version name.")
        d = ARTIFACT_DIR / version
        if not (d / "model.joblib").exists():
            raise FileNotFoundError(f"No model.joblib found for version '{version}'.")

        self.version = version
        self.model = joblib.load(d / "model.joblib")
        self.meta = json.loads((d / "metadata.json").read_text())
        self.schema = json.loads((d / "schema.json").read_text())

        self.features: List[str] = self.meta["features"]
        self.numeric: List[str] = self.meta["numeric_features"]
        self.categorical: List[str] = self.meta["categorical_features"]
        self.threshold = float(self.meta["threshold"])
        self.labels: Dict[str, Dict[str, str]] = self.schema["labels"]
        self.ranges: Dict[str, List[float]] = self.schema["numeric_ranges"]
        self.excluded = set(self.schema.get("excluded_sensitive", []))

    def env_info(self) -> Dict[str, Any]:
        trained = self.meta.get("sklearn")
        return {"sklearn_trained": trained, "sklearn_runtime": sklearn.__version__,
                "sklearn_match": trained == sklearn.__version__}

    def public_schema(self) -> Dict[str, Any]:
        return {
            "model_version": self.version,
            "numeric": self.numeric,
            "categorical": self.categorical,
            "labels": {c: self.labels[c] for c in self.categorical},
            "hard_limits": {f: list(HARD_LIMITS[f]) for f in self.numeric if f in HARD_LIMITS},
            "training_ranges": {f: self.ranges.get(f) for f in self.numeric},
            "threshold": self.threshold,
            "excluded_fields": sorted(self.excluded),
            "notice": NOTICE,
        }

    def validate(self, payload: Any):
        if not isinstance(payload, dict):
            raise ValidationFailed(["Request body must be a JSON object."])
        errors: List[str] = []
        warnings: List[str] = []
        clean: Dict[str, Any] = {}

        ignored = sorted(k for k in payload if k in self.excluded)
        for k in payload:
            if k not in self.features and k not in self.excluded:
                errors.append(f"Unknown field '{str(k)[:40]}'.")

        for f in self.numeric:
            if f not in payload:
                errors.append(f"Missing required field '{f}' (number).")
                continue
            v = payload[f]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                errors.append(f"Field '{f}' must be a finite number.")
                continue
            lo, hi = HARD_LIMITS.get(f, (-math.inf, math.inf))
            if not lo <= v <= hi:
                errors.append(f"Field '{f}' must be between {lo} and {hi}.")
                continue
            if f in WHOLE_NUMBER_FIELDS and v != int(v):
                errors.append(f"Field '{f}' must be a whole number.")
                continue
            clean[f] = float(v)
            r = self.ranges.get(f)
            if r and not (r[0] <= v <= r[1]):
                warnings.append(f"'{f}' is outside the range seen in training "
                                f"({r[0]:g} to {r[1]:g}); the score may be unreliable.")

        for f in self.categorical:
            allowed = sorted(self.labels[f])
            if f not in payload:
                errors.append(f"Missing required field '{f}' (one of {allowed}).")
                continue
            v = payload[f]
            if not (isinstance(v, str) and v in self.labels[f]):
                errors.append(f"Field '{f}' must be one of {allowed}.")
                continue
            clean[f] = v

        if errors:
            raise ValidationFailed(errors)
        return clean, warnings, ignored

    def predict(self, payload: Any) -> Dict[str, Any]:
        clean, warnings, ignored = self.validate(payload)
        row = pd.DataFrame([clean])[self.features]
        score = float(self.model.predict_proba(row)[0, 1])
        return {
            "risk_score": round(score, 4),
            "risk_class": "high" if score >= self.threshold else "lower",
            "threshold": self.threshold,
            "model_version": self.version,
            "warnings": warnings,
            "ignored_fields": ignored,
            "notice": NOTICE,
        }