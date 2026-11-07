import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.core import ModelBundle, ValidationFailed

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")  # optional: protects POST /reload


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("credit_api")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(LOG_DIR / "api.log", maxBytes=1_000_000, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning("File logging disabled: %s", e)
    return logger


log = setup_logging()


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.counts = defaultdict(int)
        self.score_sum = 0.0
        self.latency_sum_ms = 0.0

    def inc(self, key: str, n: int = 1):
        with self._lock:
            self.counts[key] += n

    def add_score(self, s: float):
        with self._lock:
            self.score_sum += s

    def add_latency(self, ms: float):
        with self._lock:
            self.latency_sum_ms += ms

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            c = dict(self.counts)
            n_pred = c.get("predictions_total", 0)
            n_req = c.get("requests_total", 0)
            return {
                "uptime_seconds": round(time.time() - self.started_at),
                "counts": c,
                "mean_risk_score": round(self.score_sum / n_pred, 4) if n_pred else None,
                "high_risk_share": round(c.get("risk_high", 0) / n_pred, 4) if n_pred else None,
                "mean_latency_ms": round(self.latency_sum_ms / n_req, 2) if n_req else None,
            }


stats = Stats()

EXAMPLE = {
    "status_checking": "A11", "duration_months": 24, "credit_history": "A32", "purpose": "A43",
    "credit_amount": 3000, "savings": "A61", "employment_since": "A73", "installment_rate": 3,
    "other_debtors": "A101", "residence_since": 2, "property": "A121", "age": 35,
    "other_installment_plans": "A143", "housing": "A152", "existing_credits": 1,
    "job": "A173", "num_dependents": 1, "telephone": "A191",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.bundle = ModelBundle()
    except Exception:
        log.exception("Could not load model artifact at startup")
        raise
    b = app.state.bundle
    log.info("Loaded model version=%s model=%s", b.version, b.meta.get("model"))
    if not b.env_info()["sklearn_match"]:
        log.warning("scikit-learn version mismatch: %s", b.env_info())
    yield


app = FastAPI(title="Loan Default Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        stats.inc("requests_total")
        stats.inc("errors_500")
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
    ms = (time.perf_counter() - start) * 1000
    stats.inc("requests_total")
    stats.inc(f"status_{response.status_code}")
    stats.add_latency(ms)
    log.info("%s %s -> %s %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(ValidationFailed)
async def on_validation_failed(request: Request, exc: ValidationFailed):
    stats.inc("errors_validation")
    log.warning("Validation failed: %s", "; ".join(exc.errors))
    return JSONResponse(status_code=422, content={"detail": "Invalid input.", "errors": exc.errors})


@app.exception_handler(RequestValidationError)
async def on_bad_body(request: Request, exc: RequestValidationError):
    stats.inc("errors_validation")
    log.warning("Malformed request body on %s", request.url.path)
    return JSONResponse(status_code=422, content={
        "detail": "Invalid input.", "errors": ["Request body must be a valid JSON object."]})


@app.get("/health")
def health(request: Request):
    b: ModelBundle = request.app.state.bundle
    return {"status": "ok", "model_version": b.version, "model": b.meta.get("model"),
            "environment": b.env_info()}


@app.get("/schema")
def schema(request: Request):
    return request.app.state.bundle.public_schema()


@app.post("/predict")
def predict(request: Request, payload: Dict[str, Any] = Body(..., examples=[EXAMPLE])):
    result = request.app.state.bundle.predict(payload)
    stats.inc("predictions_total")
    stats.inc(f"risk_{result['risk_class']}")
    stats.add_score(result["risk_score"])
    if result["warnings"]:
        stats.inc("predictions_with_warnings")
    return result


@app.get("/metrics")
def metrics():
    return stats.snapshot()


@app.post("/reload")
def reload_model(request: Request, version: Optional[str] = None,
                 x_admin_token: Optional[str] = Header(default=None)):
    """Load a new artifact version without restarting (defaults to artifacts/latest.txt)."""
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    try:
        if version is None:
            os.environ.pop("MODEL_VERSION", None)  # force a fresh read of latest.txt
        new = ModelBundle(version)
    except Exception as e:
        stats.inc("errors_reload")
        log.error("Model reload failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Could not load model: {e}")
    old = request.app.state.bundle.version
    request.app.state.bundle = new
    log.info("Model reloaded: %s -> %s", old, new.version)
    return {"status": "reloaded", "previous_version": old, "model_version": new.version}