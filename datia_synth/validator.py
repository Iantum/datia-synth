"""
datia_synth.validator
---------------------
Algoritmia de validación del dato sintético (T2.1 — Proyecto DatIA).

Métricas implementadas:
  - Distancia Wasserstein (variables numéricas)        objetivo W1 < 0.1
  - KL-divergence (variables categóricas)              objetivo KL < 0.05
  - Preservación de correlaciones ΔR (Frobenius)       objetivo ΔR < 0.08
  - Distancia PCA geométrica (centroide real vs sint.)
  - Consistencia de perfiles interseccionales (APSA/FAGA/GENERAL)

Score final compuesto:
  25% Wasserstein + 20% PCA + 25% KL + 15% correlaciones + 15% perfil
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy as kl_entropy, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from typing import Dict


# --------------------------------------------------------------------------- #
#  Clasificador de perfiles interseccionales                                   #
# --------------------------------------------------------------------------- #

def get_profile_label(row: pd.Series) -> str:
    """Clasifica un registro en APSA / FAGA / INTERSECTIONAL / GENERAL."""
    divs = row.get("diversity_types", [])
    if isinstance(divs, str):
        divs = divs.split(",") if divs else []

    is_apsa = (
        (isinstance(divs, list) and any(d not in ("NONE", "") for d in divs))
        or row.get("autonomy_level") in ("DEPENDENT", "NEEDS_ASSISTANCE")
        or row.get("main_mode") == "ADAPTED_TRANSPORT"
    )
    is_faga = (
        row.get("geo_zone_type") == "INFORMAL_SETTLEMENT"
        or row.get("main_mode") == "INFORMAL_TRANSPORT"
    )

    if is_apsa and is_faga:
        return "INTERSECTIONAL"
    if is_apsa:
        return "APSA"
    if is_faga:
        return "FAGA"
    return "GENERAL"


# --------------------------------------------------------------------------- #
#  Funciones de métricas individuales                                          #
# --------------------------------------------------------------------------- #

def _preprocess_for_pca(
    df_real: pd.DataFrame, df_synth: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_r = df_real.copy()
    df_s = df_synth.copy()
    df_r["_origin"] = 0
    df_s["_origin"] = 1
    combined = pd.concat([df_r, df_s], ignore_index=True)

    for col in combined.columns:
        if combined[col].apply(lambda x: isinstance(x, list)).any():
            combined[col] = combined[col].apply(lambda x: str(x))

    le = LabelEncoder()
    for col in combined.select_dtypes(include=["object", "category", "bool"]).columns:
        combined[col] = le.fit_transform(combined[col].astype(str))

    real_proc = combined[combined["_origin"] == 0].drop("_origin", axis=1)
    synth_proc = combined[combined["_origin"] == 1].drop("_origin", axis=1)
    return real_proc, synth_proc


def wasserstein_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Score Wasserstein normalizado [0,1] sobre columnas numéricas."""
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 1.0
    scores = []
    for col in numeric_cols:
        d = wasserstein_distance(df_real[col].fillna(0), df_synth[col].fillna(0))
        rango = df_real[col].max() - df_real[col].min()
        if rango == 0:
            rango = 1.0
        scores.append(max(0.0, 1.0 - d / rango))
    return float(np.mean(scores))


def kl_divergence_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Score KL-divergence [0,1] sobre columnas categóricas. KL=0 → 1.0."""
    cat_cols = df_real.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return 1.0
    eps = 1e-10
    col_scores = []
    for col in cat_cols:
        all_vals = pd.concat([df_real[col], df_synth[col]]).dropna().unique()
        p = np.array([df_real[col].value_counts().get(v, 0) for v in all_vals], dtype=float) + eps
        q = np.array([df_synth[col].value_counts().get(v, 0) for v in all_vals], dtype=float) + eps
        p /= p.sum()
        q /= q.sum()
        kl = float(kl_entropy(p, q))
        col_scores.append(max(0.0, 1.0 - kl))
    return float(np.mean(col_scores))


def correlation_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Score de preservación de correlaciones [0,1]. ΔR < 0.08 → score ≈ 1."""
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return 1.0
    try:
        r_real = np.corrcoef(df_real[numeric_cols].fillna(0).T)
        r_synth = np.corrcoef(df_synth[numeric_cols].fillna(0).T)
        delta_r = np.linalg.norm(r_real - r_synth, "fro")
        n = len(numeric_cols)
        return max(0.0, 1.0 - delta_r / (2.0 * n))
    except Exception:
        return 0.5


def geometric_pca_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Distancia PCA de centroide real vs. sintético normalizada a [0,1]."""
    try:
        real_proc, synth_proc = _preprocess_for_pca(df_real, df_synth)
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_proc)
        synth_scaled = scaler.transform(synth_proc)
        n_comps = min(2, real_proc.shape[1])
        pca = PCA(n_components=n_comps)
        real_pca = pca.fit_transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)
        dist = np.linalg.norm(np.mean(real_pca, axis=0) - np.mean(synth_pca, axis=0))
        return max(0.0, 1.0 - dist)
    except Exception:
        return 0.0


def ks_test_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Kolmogorov-Smirnov test sobre columnas numéricas. Score = media de (1 - KS_statistic)."""
    numeric_cols = df_real.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return 1.0
    scores = []
    for col in numeric_cols:
        stat, _ = ks_2samp(df_real[col].dropna(), df_synth[col].dropna())
        scores.append(max(0.0, 1.0 - float(stat)))
    return float(np.mean(scores))


# Reglas semánticas imposibles (campo → valores que NO pueden coexistir con otros)
_SEMANTIC_RULES = [
    # Transporte adaptado solo compatible con perfiles de dependencia
    lambda r: not (
        r.get("main_mode") == "ADAPTED_TRANSPORT"
        and r.get("autonomy_level") == "INDEPENDENT"
    ),
    # Zona informal solo para perfiles FAGA o INTERSECTIONAL
    lambda r: not (
        r.get("geo_zone_type") == "INFORMAL_SETTLEMENT"
        and r.get("diversity_types") not in (None, [], ["NONE"], "NONE", "")
        and r.get("autonomy_level") == "INDEPENDENT"
        # Permitir FAGA (zona informal) con autonomía independiente si no es APSA
    ),
    # Trayectos negativos o absurdos (>= 0)
    lambda r: (r.get("trips_per_week") is None or float(r.get("trips_per_week", 0) or 0) >= 0),
]


def semantic_validity_score(df_synth: pd.DataFrame) -> float:
    """
    Tasa de registros sintéticos sin errores semánticos.
    Objetivo: Ssem > 0.99 (tasa de error < 1%).
    Retorna score en [0, 1].
    """
    if df_synth.empty:
        return 1.0
    valid = 0
    for _, row in df_synth.iterrows():
        if all(rule(row) for rule in _SEMANTIC_RULES):
            valid += 1
    return round(valid / len(df_synth), 4)


def umap_lnp_score(df_real: pd.DataFrame, df_synth: pd.DataFrame, k: int = 15) -> float:
    """Local Neighborhood Preservation via UMAP. Objetivo: LNP ≥ 0.7."""
    try:
        import umap  # type: ignore
    except ImportError:
        return 0.0

    numeric_cols = df_real.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2 or len(df_real) < max(k + 1, 15):
        return 0.0

    try:
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(df_real[numeric_cols].fillna(0))
        synth_scaled = scaler.transform(df_synth[numeric_cols].fillna(0))

        reducer = umap.UMAP(n_components=2, n_neighbors=k, random_state=42)
        real_emb = reducer.fit_transform(real_scaled)
        synth_emb = reducer.transform(synth_scaled)

        # k-NN en espacio original
        nn_orig = NearestNeighbors(n_neighbors=k + 1).fit(real_scaled)
        _, idx_orig = nn_orig.kneighbors(real_scaled)

        # k-NN en espacio UMAP
        nn_emb = NearestNeighbors(n_neighbors=k + 1).fit(real_emb)
        _, idx_emb = nn_emb.kneighbors(real_emb)

        # Fracción media de vecinos preservados (excluye el propio punto)
        preserved = []
        for i in range(len(real_scaled)):
            orig_neighbors = set(idx_orig[i, 1:])
            emb_neighbors = set(idx_emb[i, 1:])
            preserved.append(len(orig_neighbors & emb_neighbors) / k)

        return float(np.mean(preserved))
    except Exception:
        return 0.0


def tda_homology_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Wasserstein entre diagramas de persistencia H0+H1. Objetivo: score > 0.7."""
    try:
        from ripser import ripser as compute_ripser  # type: ignore
        from persim import wasserstein as persim_wasserstein  # type: ignore
    except ImportError:
        return 0.0

    numeric_cols = df_real.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return 0.0

    try:
        scaler = StandardScaler()
        # Submuestreo máximo 500 puntos para escalar con ripser
        max_pts = 500
        r_data = scaler.fit_transform(df_real[numeric_cols].fillna(0))
        s_data = scaler.transform(df_synth[numeric_cols].fillna(0))
        if len(r_data) > max_pts:
            idx = np.random.choice(len(r_data), max_pts, replace=False)
            r_data = r_data[idx]
        if len(s_data) > max_pts:
            idx = np.random.choice(len(s_data), max_pts, replace=False)
            s_data = s_data[idx]

        dgm_real = compute_ripser(r_data, maxdim=1)["dgms"]
        dgm_synth = compute_ripser(s_data, maxdim=1)["dgms"]

        scores = []
        for dim in range(min(2, len(dgm_real))):
            d_r = dgm_real[dim]
            d_s = dgm_synth[dim]
            # Eliminar puntos en infinito
            d_r = d_r[np.isfinite(d_r[:, 1])] if len(d_r) > 0 else d_r
            d_s = d_s[np.isfinite(d_s[:, 1])] if len(d_s) > 0 else d_s
            if len(d_r) == 0 and len(d_s) == 0:
                scores.append(1.0)
                continue
            dist = persim_wasserstein(d_r, d_s, matching=False)
            scores.append(float(np.exp(-dist)))

        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0


def profile_consistency_score(df_real: pd.DataFrame, df_synth: pd.DataFrame) -> float:
    """Preservación de distribución de perfiles APSA/FAGA/INTERSECTIONAL/GENERAL."""
    real_p = df_real.apply(get_profile_label, axis=1).value_counts(normalize=True)
    synth_p = df_synth.apply(get_profile_label, axis=1).value_counts(normalize=True)
    all_labels = set(real_p.index) | set(synth_p.index)
    diffs = [abs(real_p.get(lbl, 0) - synth_p.get(lbl, 0)) for lbl in all_labels]
    return max(0.0, 1.0 - float(np.sum(diffs)))


# --------------------------------------------------------------------------- #
#  Clase pública                                                               #
# --------------------------------------------------------------------------- #

class DataQualityValidator:
    """
    Validador de calidad del dato sintético (E1.2 / E2.1 — DatIA).

    Uso::

        validator = DataQualityValidator()
        report = validator.validate(df_real, df_synthetic)
        print(report["final_score"])   # 0.0 – 1.0
    """

    WEIGHTS = {
        "wasserstein": 0.15,
        "ks":          0.10,
        "kl":          0.15,
        "geometric":   0.12,
        "correlation": 0.12,
        "profile":     0.12,
        "semantic":    0.09,
        "umap_lnp":    0.10,
        "tda":         0.05,
    }

    def validate(
        self,
        df_real: pd.DataFrame,
        df_synth: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Valida la calidad del conjunto sintético respecto al real.

        Parámetros
        ----------
        df_real : pd.DataFrame   — datos reales de referencia
        df_synth: pd.DataFrame  — datos sintéticos a evaluar

        Retorna
        -------
        dict con claves:
            final_score, wasserstein_score, kl_divergence_score,
            correlation_score, geometric_pca_score, profile_consistency_score
        """
        if df_synth.empty:
            return {k: 0.0 for k in (
                "final_score", "wasserstein_score", "ks_test_score",
                "kl_divergence_score", "correlation_score", "geometric_pca_score",
                "profile_consistency_score", "semantic_validity_score",
                "umap_lnp_score", "tda_homology_score",
            )}

        def _safe(v: float) -> float:
            """Sustituye NaN/Inf por 0.0 para que la búsqueda binaria no se rompa."""
            import math
            return 0.0 if (v is None or math.isnan(v) or math.isinf(v)) else float(v)

        w   = _safe(wasserstein_score(df_real, df_synth))
        ks  = _safe(ks_test_score(df_real, df_synth))
        kl  = _safe(kl_divergence_score(df_real, df_synth))
        cr  = _safe(correlation_score(df_real, df_synth))
        gm  = _safe(geometric_pca_score(df_real, df_synth))
        pr  = _safe(profile_consistency_score(df_real, df_synth))
        sem = _safe(semantic_validity_score(df_synth))
        ul  = _safe(umap_lnp_score(df_real, df_synth))
        tda = _safe(tda_homology_score(df_real, df_synth))

        W = self.WEIGHTS
        final = _safe(
            w   * W["wasserstein"] +
            ks  * W["ks"] +
            kl  * W["kl"] +
            gm  * W["geometric"] +
            cr  * W["correlation"] +
            pr  * W["profile"] +
            sem * W["semantic"] +
            ul  * W["umap_lnp"] +
            tda * W["tda"]
        )

        return {
            "final_score":               round(final, 4),
            "wasserstein_score":         round(w,   4),
            "ks_test_score":             round(ks,  4),
            "kl_divergence_score":       round(kl,  4),
            "correlation_score":         round(cr,  4),
            "geometric_pca_score":       round(gm,  4),
            "profile_consistency_score": round(pr,  4),
            "semantic_validity_score":   round(sem, 4),
            "umap_lnp_score":            round(ul,  4),
            "tda_homology_score":        round(tda, 4),
        }
