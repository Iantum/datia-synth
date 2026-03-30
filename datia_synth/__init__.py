"""
datia_synth — Librería open source de generación y validación de datos sintéticos
==================================================================================

Entregables E1.2 y E2.1 del Proyecto DatIA (Movilidad, Salud y Acción Social).
Desarrollado por INFORTIC para CEU-UCH / FAGA / APSA.

Uso rápido::

    from datia_synth import DatiaSynthesizer, DataQualityValidator

    # Generación
    synth = DatiaSynthesizer(quality_threshold=0.75)
    result = synth.generate(my_records)

    # Validación independiente
    validator = DataQualityValidator()
    report = validator.validate(df_real, df_synthetic)
    print(report["final_score"])
"""

from .generator import DatiaSynthesizer
from .validator import DataQualityValidator
from .crypto import PostQuantumSigner, SphincsBackupSigner, KyberKeyEncapsulator

__version__ = "1.0.0"
__all__ = [
    "DatiaSynthesizer",
    "DataQualityValidator",
    "PostQuantumSigner",
    "SphincsBackupSigner",
    "KyberKeyEncapsulator",
]
