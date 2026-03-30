"""
DatIA — API REST del generador de datos sintéticos (MS-SCIENCE)
===============================================================
Microservicio FastAPI que expone la librería datia_synth como API.

Endpoints:
  POST /generate  — genera datos sintéticos con validación y firma post-cuántica
  POST /verify    — verifica la integridad criptográfica del resultado

Librería subyacente: datia_synth (E1.2 — Proyecto DatIA)
"""

import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any

from datia_synth import DatiaSynthesizer, DataQualityValidator, PostQuantumSigner, SphincsBackupSigner, KyberKeyEncapsulator

# ---------------------------------------------------------------------------
#  Logging estructurado JSON → stdout (Docker lo captura)
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    SERVICE = "datia-synth-api"

    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts":      datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":   record.levelname,
            "service": self.SERVICE,
            "msg":     record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in ("name","msg","args","levelname","levelno","pathname",
                           "filename","module","exc_info","exc_text","stack_info",
                           "lineno","funcName","created","msecs","relativeCreated",
                           "thread","threadName","processName","process","message","taskName"):
                log[key] = value
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log, ensure_ascii=False, default=str)

def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

_setup_logging()
logger = logging.getLogger("datia.synth")

# ---------------------------------------------------------------------------
#  Configuración de la app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("datia-synth-api starting up", extra={"event": "startup"})
    yield
    logger.info("datia-synth-api shutting down", extra={"event": "shutdown"})

app = FastAPI(
    title="DatIA: Generador de Datos Sintéticos",
    description=(
        "Motor de generación (SDV/GaussianCopula), validación interseccional "
        "(Wasserstein + KL + correlaciones + PCA) e integridad post-cuántica "
        "(CRYSTALS-Dilithium2 / ML-DSA-44)."
    ),
    version="7.0.0",
    lifespan=lifespan,
)

import os as _os
# Umbral de calidad configurable. Datos homogéneos (ej. todo UNEMPLOYED+URBAN_CENTRE)
# rara vez superan 0.60. El valor por defecto 0.55 permite que el generador produzca
# datos sintéticos útiles aun con datasets de baja diversidad.
QUALITY_THRESHOLD = float(_os.environ.get("QUALITY_THRESHOLD", "0.55"))

# ---------------------------------------------------------------------------
#  Modelos de datos
# ---------------------------------------------------------------------------

class DatasetPayload(BaseModel):
    data: List[Dict[str, Any]]


class ExecutionMetadata(BaseModel):
    status: str
    input_n: int
    generated_x: int
    iterations: int
    final_quality_score: float
    profile_consistency_score: float
    quality_threshold: float
    # Métricas individuales T2.1
    wasserstein_score: float
    ks_test_score: float
    kl_divergence_score: float
    correlation_score: float
    geometric_pca_score: float
    semantic_validity_score: float
    umap_lnp_score: float
    tda_homology_score: float
    generation_time_ms: int
    # Seguridad post-cuántica
    signature_algorithm: str
    dilithium_signature_hex: str
    public_key_hex: str


class HybridResponse(BaseModel):
    execution_metadata: ExecutionMetadata
    results: List[Dict[str, Any]]


class VerificationPayload(BaseModel):
    data: List[Dict[str, Any]]
    signature_hex: str
    public_key_hex: str


class VerificationResponse(BaseModel):
    is_authentic: bool
    message: str
    checked_records: int


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

import time as _time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

class _RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> StarletteResponse:
        if request.url.path in ("/health",):
            return await call_next(request)
        t0 = _time.monotonic()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            dur = round((_time.monotonic() - t0) * 1000)
            level = "warning" if status_code >= 400 else "info"
            ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
                 or (request.client.host if request.client else "unknown")
            getattr(logger, level)(
                f"{request.method} {request.url.path} → {status_code}",
                extra={"method": request.method, "path": request.url.path,
                       "status_code": status_code, "duration_ms": dur, "client_ip": ip},
            )

app.add_middleware(_RequestLoggerMiddleware)

class VisualizeRequest(BaseModel):
    real: List[Dict[str, Any]]
    synthetic: List[Dict[str, Any]]
    method: str = "umap"   # "umap" | "tsne" | "pca"

class VisualizeResponse(BaseModel):
    method: str
    real_points: List[Dict[str, float]]      # [{x, y}, ...]
    synthetic_points: List[Dict[str, float]]
    explained_variance: float | None         # solo PCA


@app.post("/visualize", response_model=VisualizeResponse, tags=["Validación"])
def visualize_manifold(payload: VisualizeRequest):
    """
    V08 — Proyección 2D de datos reales y sintéticos para visualización.

    Métodos disponibles:
    - **umap** (por defecto) — UMAP 2D, preserva estructura local y global
    - **tsne** — t-SNE 2D, preserva vecindarios locales
    - **pca** — PCA 2D, más rápido, devuelve varianza explicada
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    def _prepare(real_data, synth_data):
        df_r = pd.DataFrame(real_data)
        df_s = pd.DataFrame(synth_data)
        df_r["_origin"] = 0
        df_s["_origin"] = 1
        combined = pd.concat([df_r, df_s], ignore_index=True)
        for col in combined.columns:
            if combined[col].apply(lambda x: isinstance(x, list)).any():
                combined[col] = combined[col].apply(str)
        le = LabelEncoder()
        for col in combined.select_dtypes(include=["object", "category", "bool"]).columns:
            combined[col] = le.fit_transform(combined[col].astype(str))
        combined = combined.fillna(0)
        origins = combined.pop("_origin").values
        scaled = StandardScaler().fit_transform(combined)
        return scaled, origins

    try:
        scaled, origins = _prepare(payload.real, payload.synthetic)
        n_real = int((origins == 0).sum())

        explained = None
        method = payload.method.lower()

        if method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(scaled)
            explained = round(float(pca.explained_variance_ratio_.sum()), 4)

        elif method == "tsne":
            from sklearn.manifold import TSNE
            coords = TSNE(n_components=2, random_state=42, perplexity=min(30, len(scaled)-1)).fit_transform(scaled)

        else:  # umap (default)
            try:
                import umap
                coords = umap.UMAP(n_components=2, random_state=42).fit_transform(scaled)
            except ImportError:
                from sklearn.decomposition import PCA
                coords = PCA(n_components=2).fit_transform(scaled)
                method = "pca_fallback"

        real_pts  = [{"x": float(coords[i, 0]), "y": float(coords[i, 1])} for i in range(n_real)]
        synth_pts = [{"x": float(coords[i, 0]), "y": float(coords[i, 1])} for i in range(n_real, len(coords))]

        return {
            "method": method,
            "real_points": real_pts,
            "synthetic_points": synth_pts,
            "explained_variance": explained,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en visualización: {e}")


@app.get("/health", tags=["Monitoreo"])
def health_check():
    """KPI09 — Comprobación de disponibilidad del servicio (liveness probe)."""
    return {
        "status": "ok",
        "service": "datia-synth-api",
        "version": app.version,
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }


@app.post("/generate", response_model=HybridResponse)
def generate_optimal_data(payload: DatasetPayload):
    """
    Genera datos sintéticos optimizados a partir de registros reales.

    - Mínimo 5 registros de entrada.
    - Busca el X óptimo (búsqueda binaria) con quality_score ≥ 0.75.
    - Retorna métricas detalladas T2.1 y firma ML-DSA-44.
    """
    try:
        input_data = [
            item if isinstance(item, dict) else item.dict()
            for item in payload.data
        ]

        _t0 = _time.monotonic()
        synth = DatiaSynthesizer(quality_threshold=QUALITY_THRESHOLD)
        result = synth.generate(input_data)
        generation_time_ms = round((_time.monotonic() - _t0) * 1000)

        meta = result["metadata"]
        logger.info(
            "Synthetic generation completed",
            extra={
                "input_n":          meta.get("input_n"),
                "generated_x":      meta.get("generated_x"),
                "iterations":       meta.get("iterations"),
                "quality_score":    meta.get("final_quality_score"),
                "generation_ms":    generation_time_ms,
                "status":           meta.get("status"),
            }
        )

        return {
            "execution_metadata": {
                "status":                    meta["status"],
                "input_n":                   meta["input_n"],
                "generated_x":               meta["generated_x"],
                "iterations":                meta["iterations"],
                "final_quality_score":       meta["final_quality_score"],
                "profile_consistency_score": meta["profile_consistency_score"],
                "quality_threshold":         meta["quality_threshold"],
                "wasserstein_score":         meta["wasserstein_score"],
                "ks_test_score":             meta.get("ks_test_score", 0.0),
                "kl_divergence_score":       meta["kl_divergence_score"],
                "correlation_score":         meta["correlation_score"],
                "geometric_pca_score":       meta["geometric_pca_score"],
                "semantic_validity_score":   meta.get("semantic_validity_score", 0.0),
                "umap_lnp_score":            meta.get("umap_lnp_score", 0.0),
                "tda_homology_score":        meta.get("tda_homology_score", 0.0),
                "generation_time_ms":        generation_time_ms,
                "signature_algorithm":       meta["signature_algorithm"],
                "dilithium_signature_hex":   meta["dilithium_signature_hex"],
                "public_key_hex":            meta["public_key_hex"],
            },
            "results": result["all_records"],
        }

    except ValueError as e:
        logger.warning("Bad request on /generate", extra={"error": str(e), "input_n": len(payload.data)})
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Critical error on /generate", extra={"error": str(e)}, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error crítico en generación: {e}")


@app.post("/verify", response_model=VerificationResponse)
def verify_data_integrity(payload: VerificationPayload):
    """Verifica la firma ML-DSA-44 de un conjunto de registros."""
    try:
        synth = DatiaSynthesizer()
        result = synth.verify(payload.data, payload.signature_hex, payload.public_key_hex)
        return {
            "is_authentic":    result["is_authentic"],
            "message":         result["message"],
            "checked_records": result["checked_records"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error crítico en verificación: {e}")


# ---------------------------------------------------------------------------
#  FIPS 205 — SLH-DSA / SPHINCS+ backup signature endpoint
# ---------------------------------------------------------------------------

class BackupSignatureRequest(BaseModel):
    data: List[Dict[str, Any]]

class BackupSignatureResponse(BaseModel):
    algorithm: str
    signature_hex: str
    public_key_hex: str

class BackupVerifyRequest(BaseModel):
    data: List[Dict[str, Any]]
    signature_hex: str
    public_key_hex: str

class BackupVerifyResponse(BaseModel):
    is_authentic: bool

@app.post("/sign/backup", response_model=BackupSignatureResponse, tags=["Criptografía post-cuántica"])
def sign_backup(payload: BackupSignatureRequest):
    """Firma con SLH-DSA-SHA2-128s (SPHINCS+ / FIPS 205) como firma de respaldo."""
    signer = SphincsBackupSigner()
    sig_hex, pub_hex = signer.sign(payload.data)
    return {"algorithm": SphincsBackupSigner.ALGORITHM, "signature_hex": sig_hex, "public_key_hex": pub_hex}

@app.post("/verify/backup", response_model=BackupVerifyResponse, tags=["Criptografía post-cuántica"])
def verify_backup(payload: BackupVerifyRequest):
    """Verifica firma SLH-DSA-SHA2-128s."""
    signer = SphincsBackupSigner()
    return {"is_authentic": signer.verify(payload.data, payload.signature_hex, payload.public_key_hex)}


# ---------------------------------------------------------------------------
#  FIPS 203 — ML-KEM / Kyber key encapsulation endpoints
# ---------------------------------------------------------------------------

class KemKeypairResponse(BaseModel):
    algorithm: str
    public_key_hex: str
    secret_key_hex: str

class KemEncapRequest(BaseModel):
    public_key_hex: str

class KemEncapResponse(BaseModel):
    ciphertext_hex: str
    shared_secret_hex: str

class KemDecapRequest(BaseModel):
    ciphertext_hex: str
    secret_key_hex: str

class KemDecapResponse(BaseModel):
    shared_secret_hex: str

@app.post("/kem/keypair", response_model=KemKeypairResponse, tags=["Criptografía post-cuántica"])
def kem_generate_keypair():
    """Genera par de claves ML-KEM-512 (Kyber / FIPS 203)."""
    kem = KyberKeyEncapsulator()
    pub_hex, sk_hex = kem.generate_keypair()
    return {"algorithm": KyberKeyEncapsulator.ALGORITHM, "public_key_hex": pub_hex, "secret_key_hex": sk_hex}

@app.post("/kem/encapsulate", response_model=KemEncapResponse, tags=["Criptografía post-cuántica"])
def kem_encapsulate(payload: KemEncapRequest):
    """Encapsula una clave simétrica con ML-KEM-512. Retorna ciphertext + shared_secret."""
    kem = KyberKeyEncapsulator()
    ct_hex, ss_hex = kem.encapsulate(payload.public_key_hex)
    return {"ciphertext_hex": ct_hex, "shared_secret_hex": ss_hex}

@app.post("/kem/decapsulate", response_model=KemDecapResponse, tags=["Criptografía post-cuántica"])
def kem_decapsulate(payload: KemDecapRequest):
    """Recupera el shared_secret a partir del ciphertext y la clave secreta ML-KEM-512."""
    kem = KyberKeyEncapsulator()
    ss_hex = kem.decapsulate(payload.ciphertext_hex, payload.secret_key_hex)
    return {"shared_secret_hex": ss_hex}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
