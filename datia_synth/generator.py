"""
datia_synth.generator
----------------------
Motor de generación de datos sintéticos (E1.2 — Proyecto DatIA).

Algoritmo:
  1. Aplanamiento de columnas lista → string (compatibilidad SDV)
  2. Entrenamiento GaussianCopulaSynthesizer (SDV)
  3. Búsqueda binaria del X óptimo que maximiza calidad ≥ umbral
  4. Firma post-cuántica ML-DSA-44 del resultado

Referencia: T1.2 / T2.1 Memoria del Proyecto DatIA (2025-2026)
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List, Optional

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

from .validator import DataQualityValidator
from .crypto import PostQuantumSigner


QUALITY_THRESHOLD_DEFAULT = 0.75


class DatiaSynthesizer:
    """
    Generador de datos sintéticos con validación de calidad interseccional
    y firma post-cuántica (ML-DSA-44).

    Uso::

        synth = DatiaSynthesizer(quality_threshold=0.75)
        result = synth.generate(records)
        print(result["metadata"]["final_quality_score"])
        print(len(result["synthetic_records"]))

    Parámetros
    ----------
    quality_threshold : float
        Score mínimo aceptable [0,1]. Por defecto 0.75.
    """

    def __init__(self, quality_threshold: float = QUALITY_THRESHOLD_DEFAULT):
        self.quality_threshold = quality_threshold
        self._validator = DataQualityValidator()
        self._signer = PostQuantumSigner()

    # ------------------------------------------------------------------ #
    #  API pública                                                         #
    # ------------------------------------------------------------------ #

    def generate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera datos sintéticos a partir de registros reales.

        Parámetros
        ----------
        records : list of dict — registros reales de entrada (mín. 5)

        Retorna
        -------
        dict con claves:
            metadata         — ExecutionMetadata (scores, iteraciones, firma…)
            real_records     — registros reales etiquetados is_synthetic=False
            synthetic_records— registros sintéticos etiquetados is_synthetic=True
            all_records      — combinación de ambos (firmada)
        """
        if len(records) < 5:
            raise ValueError("Se requieren al menos 5 registros reales.")

        df_real = pd.DataFrame(records)
        list_cols = self._flatten_lists(df_real)

        # Eliminar columnas que confunden a SDV:
        # UUIDs (alta cardinalidad única), datetimes y columnas con un solo valor único
        _DROP_PATTERNS = {"uuid", "id", "created_at", "timestamp", "batch", "profile_id",
                          "pattern_id", "user_uuid", "batch_id", "direccion"}
        cols_to_drop = [
            c for c in df_real.columns
            if c.lower() in _DROP_PATTERNS
            or any(p in c.lower() for p in ("uuid", "_id", "created"))
            # Solo eliminar strings con cardinalidad 1:1 (identificadores textuales)
            # Nunca eliminar columnas numéricas aunque sean únicas (ej: birth_year)
            or (df_real[c].dtype == object and df_real[c].nunique() == len(df_real))
        ]
        df_synth_input = df_real.drop(columns=cols_to_drop, errors="ignore")
        # Eliminar filas con NaN en columnas numéricas (edad=NULL, etc.)
        # SDV no maneja NaN y destroza wasserstein/PCA/correlación
        numeric_cols = df_synth_input.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            df_synth_input = df_synth_input.dropna(subset=numeric_cols).reset_index(drop=True)

        # Entrenamiento
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_synth_input)
        # SDV detecta columnas numéricas secuenciales (ej: birth_year) como 'id'
        # y las asigna como primary_key → corregir a 'numerical' y eliminar pk
        # IMPORTANTE: hay que eliminar el PK ANTES de cambiar sdtype (update_column
        # valida contra el PK constraint y falla si el PK no es tipo 'id')
        for col in df_synth_input.select_dtypes(include="number").columns:
            if metadata.columns.get(col, {}).get("sdtype") == "id":
                if metadata.primary_key == col:
                    metadata.primary_key = None
                metadata.columns[col] = {"sdtype": "numerical"}
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(df_synth_input)

        # Búsqueda binaria (usa df_synth_input — sin UUIDs ni datetimes)
        best_df, best_metrics, iterations, status = self._binary_search(
            df_synth_input, synthesizer
        )

        # Consolidación
        df_real_out = df_real.copy()
        df_real_out["is_synthetic"] = False

        if not best_df.empty:
            best_df["is_synthetic"] = True
            df_all = pd.concat([df_real_out, best_df], ignore_index=True)
        else:
            df_all = df_real_out

        self._restore_lists(df_all, list_cols)

        all_records = df_all.to_dict(orient="records")
        syn_records = [r for r in all_records if r.get("is_synthetic")]
        real_records = [r for r in all_records if not r.get("is_synthetic")]

        # Firma post-cuántica
        sig_hex, pub_hex = self._signer.sign(all_records)

        meta = {
            "status":                  status,
            "input_n":                 len(records),
            "generated_x":             len(syn_records),
            "iterations":              iterations,
            "quality_threshold":       self.quality_threshold,
            "final_quality_score":     best_metrics.get("final_score", 0.0),
            "wasserstein_score":       best_metrics.get("wasserstein_score", 0.0),
            "kl_divergence_score":     best_metrics.get("kl_divergence_score", 0.0),
            "correlation_score":       best_metrics.get("correlation_score", 0.0),
            "geometric_pca_score":     best_metrics.get("geometric_pca_score", 0.0),
            "profile_consistency_score": best_metrics.get("profile_consistency_score", 0.0),
            "ks_test_score":           best_metrics.get("ks_test_score", 0.0),
            "semantic_validity_score": best_metrics.get("semantic_validity_score", 0.0),
            "umap_lnp_score":          best_metrics.get("umap_lnp_score", 0.0),
            "tda_homology_score":      best_metrics.get("tda_homology_score", 0.0),
            "signature_algorithm":     "ML-DSA-44 (Crystals-Dilithium2)",
            "dilithium_signature_hex": sig_hex,
            "public_key_hex":          pub_hex,
        }

        return {
            "metadata":          meta,
            "real_records":      real_records,
            "synthetic_records": syn_records,
            "all_records":       all_records,
        }

    def verify(
        self,
        records: List[Dict[str, Any]],
        signature_hex: str,
        public_key_hex: str,
    ) -> Dict[str, Any]:
        """Verifica la firma ML-DSA-44 de un conjunto de registros."""
        is_valid = self._signer.verify(records, signature_hex, public_key_hex)
        return {
            "is_authentic":   is_valid,
            "checked_records": len(records),
            "message": (
                "Firma válida. Los datos no han sido manipulados."
                if is_valid
                else "ALERTA: Firma inválida. Los datos pueden haber sido alterados."
            ),
        }

    # ------------------------------------------------------------------ #
    #  Helpers privados                                                    #
    # ------------------------------------------------------------------ #

    def _flatten_lists(self, df: pd.DataFrame) -> List[str]:
        """Aplana columnas lista a string CSV. Retorna los nombres afectados."""
        list_cols = []
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                list_cols.append(col)
                df[col] = df[col].apply(
                    lambda x: ",".join(x) if isinstance(x, list) else (x or "")
                )
        return list_cols

    def _restore_lists(self, df: pd.DataFrame, list_cols: List[str]) -> None:
        """Restaura columnas previamente aplanadas a listas."""
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: x.split(",") if isinstance(x, str) and x else []
                )

    def _binary_search(
        self,
        df_real: pd.DataFrame,
        synthesizer: GaussianCopulaSynthesizer,
    ):
        """Búsqueda binaria del X óptimo ≥ quality_threshold."""
        N = len(df_real)
        low, high = 1, N - 1
        best_df = pd.DataFrame()
        best_metrics: Dict = {}
        best_score = -1.0
        iterations = 0
        status = "FAILED"

        while low <= high:
            iterations += 1
            mid = (low + high) // 2
            if mid <= 0:
                low = 1
                if low > high:
                    break
                continue

            candidate = synthesizer.sample(num_rows=mid)
            metrics = self._validator.validate(df_real, candidate)
            score = metrics.get("final_score", 0.0)

            # Siempre trackear el mejor resultado (para reportar métricas reales)
            if score > best_score:
                best_score = score
                best_df = candidate
                best_metrics = metrics

            if score >= self.quality_threshold:
                status = "OPTIMAL_FOUND"
                low = mid + 1
            else:
                high = mid - 1

        return best_df, best_metrics, iterations, status
