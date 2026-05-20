# datia-synth

**Open-source library for synthetic data generation and validation in mobility, health, and social action domains.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FIPS 204](https://img.shields.io/badge/Post--Quantum-ML--DSA--44%20FIPS%20204-green.svg)](https://csrc.nist.gov/pubs/fips/204/final)

Developed by **INFORTIC** and **CEU-UCH** as part of [Proyecto DatIA](https://github.com/iantum/datia-synth), funded by IVACE+i (Generalitat Valenciana) under the sectoral data spaces programme.

---

## What is datia-synth?

`datia-synth` generates high-fidelity synthetic records from real mobility and health data collected from vulnerable communities (Roma community via FAGA, people with disabilities via APSA/Parkinson), while:

- **Preserving statistical properties** — Wasserstein distance, KL-divergence, KS-test
- **Preserving geometric structure** — PCA centroids, UMAP Local Neighborhood Preservation
- **Preserving topological features** — TDA persistence diagrams (H0, H1)
- **Preserving semantic validity** — domain-specific rules and intersectional profile consistency
- **Signing every batch** with post-quantum cryptography (ML-DSA-44, FIPS 204)

---

## Installation

```bash
# Core library (generation + validation)
pip install datia-synth

# With REST API (FastAPI)
pip install "datia-synth[api]"

# With post-quantum cryptography (requires liboqs)
pip install "datia-synth[crypto]"

# Full installation
pip install "datia-synth[all]"

# From source
git clone https://github.com/iantum/datia-synth.git
cd datia-synth
pip install -e ".[all]"
```

---

## Quick start

### Generate synthetic data

```python
from datia_synth import DatiaSynthesizer

records = [
    {
        "genero": "Mujer", "laboral": "Empleada", "autonomia": "INDEPENDENT",
        "medio_transporte": "PUBLIC_TRANSPORT", "trayectos": "5",
        "discapacidad": "NONE", "colectivo": "GENERAL",
        # ... more fields
    },
    # minimum 15 records recommended
]

synth = DatiaSynthesizer(quality_threshold=0.75)
result = synth.generate(records)

print(result["execution_metadata"]["status"])        # OPTIMAL_FOUND
print(result["execution_metadata"]["quality_score"]) # 0.82
print(result["execution_metadata"]["generated_x"])   # e.g. 51
print(result["synthetic_records"][:2])               # first 2 synthetic records
```

### Validate independently

```python
import pandas as pd
from datia_synth import DataQualityValidator

validator = DataQualityValidator()
metrics = validator.validate(df_real, df_synth)

# {
#   "quality_score": 0.82,
#   "wasserstein": 0.91,
#   "kl": 0.78,
#   "ks": 0.85,
#   "geometric": 0.76,
#   "correlation": 0.88,
#   "profile": 0.79,
#   "semantic": 0.92,
#   "umap_lnp": 0.72,
#   "tda": 0.68
# }
```

### Verify cryptographic signature

```python
from datia_synth import DatiaSynthesizer

synth = DatiaSynthesizer()
result = synth.verify(records, signature_hex, public_key_hex)
print(result["is_authentic"])  # True / False
```

### Post-quantum cryptography (standalone)

```python
from datia_synth import PostQuantumSigner, KyberKeyEncapsulator

# ML-DSA-44 (FIPS 204) — sign and verify
signer = PostQuantumSigner()
sig_hex, pub_key_hex = signer.sign(b"my data")
is_valid = signer.verify(b"my data", sig_hex, pub_key_hex)

# ML-KEM-512 (FIPS 203) — key encapsulation
kem = KyberKeyEncapsulator()
pub_hex, sec_hex = kem.generate_keypair()
ciphertext_hex, shared_secret_hex = kem.encapsulate(pub_hex)
recovered_secret_hex = kem.decapsulate(ciphertext_hex, sec_hex)
```

---

## REST API

```bash
# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Or with Docker
docker build -t datia-synth .
docker run -p 8001:8001 datia-synth
```

Interactive API docs available at `http://localhost:8001/docs`.

### Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/generate` | Generate synthetic data with validation and signature |
| `POST` | `/verify` | Verify ML-DSA-44 signature of a batch |
| `POST` | `/visualize` | 2D projection (UMAP/t-SNE/PCA) of real vs synthetic |
| `POST` | `/sign/backup` | Sign with SLH-DSA backup (FIPS 205) |
| `POST` | `/verify/backup` | Verify SLH-DSA signature |
| `POST` | `/kem/keypair` | Generate ML-KEM-512 keypair |
| `POST` | `/kem/encapsulate` | Encapsulate with ML-KEM public key |
| `POST` | `/kem/decapsulate` | Decapsulate with ML-KEM private key |
| `GET`  | `/health` | Health check |

---

## Quality metrics

| Metric | Weight | Target | Description |
|--------|--------|--------|-------------|
| Wasserstein (W₁) | 15% | W₁ < 0.1 | Marginal distribution distance (numerical) |
| KL-divergence | 15% | KL < 0.05 | Categorical distribution fidelity |
| KS-test | 10% | p > 0.05 | Kolmogorov-Smirnov test |
| PCA geometric | 12% | low | Centroid distance in PCA space |
| Correlation (ΔR) | 12% | ΔR < 0.08 | Frobenius norm of correlation difference |
| Profile consistency | 12% | high | APSA/FAGA intersectional profile coherence |
| Semantic validity | 9% | > 95% | Domain-specific rule compliance |
| UMAP LNP | 10% | ≥ 0.7 | Local Neighborhood Preservation |
| TDA homology | 5% | > 0.7 | Wasserstein distance between H0+H1 persistence diagrams |
| **Composite score** | **100%** | **≥ 0.75** | Weighted average |

---

## Package structure

```
datia-synth/
├── datia_synth/
│   ├── __init__.py      # Public exports
│   ├── generator.py     # DatiaSynthesizer — GaussianCopula + binary search
│   ├── validator.py     # DataQualityValidator — 9 quality dimensions
│   └── crypto.py        # PostQuantumSigner, SphincsBackupSigner, KyberKeyEncapsulator
├── main.py              # FastAPI REST API
├── pyproject.toml       # pip package configuration
├── requirements.txt     # Pinned dependencies for reproducibility
└── Dockerfile           # Container deployment
```

---

## Post-quantum cryptography

All synthetic batches are signed using **ML-DSA-44 (CRYSTALS-Dilithium, FIPS 204)**, ensuring long-term authenticity even against quantum adversaries.

| Standard | Algorithm | Purpose |
|----------|-----------|---------|
| FIPS 204 | ML-DSA-44 | Primary batch signature |
| FIPS 205 | SLH-DSA-SHA2-128s | Backup hash-based signature |
| FIPS 203 | ML-KEM-512 | Secure key encapsulation |

> **Note:** Post-quantum features require `liboqs-python`. Install with `pip install "datia-synth[crypto]"`. If unavailable, the library degrades gracefully — generation continues without cryptographic signing.

---

## Requirements

- Python 3.9+
- numpy ≥ 1.26, pandas ≥ 2.1, scipy ≥ 1.11, scikit-learn ≥ 1.4
- sdv ≥ 1.9 (synthetic data generation)
- umap-learn ≥ 0.5.6 (geometric validation)
- ripser ≥ 0.6.8, persim ≥ 0.3.7 (topological validation)
- `liboqs-python` ≥ 0.10.0 (optional, post-quantum crypto)

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Citation

If you use datia-synth in your research, please cite:

```bibtex
@software{datia_synth_2026,
  author    = {INFORTIC and CEU-UCH},
  title     = {datia-synth: Open-source synthetic data generation for mobility and health},
  year      = {2026},
  url       = {https://github.com/iantum/datia-synth},
  license   = {Apache-2.0}
}
```

---

## Acknowledgements

This library was developed as part of **Proyecto DatIA** (2025–2026), funded by:
- **IVACE+i** (Institut Valencià de Competitivitat Empresarial) — Generalitat Valenciana
- Programme: *Ayudas en materia de espacios de datos sectoriales en la Comunitat Valenciana*

Consortium: **CEU-UCH** (scientific lead) · **INFORTIC** (platform lead) · **FAGA** · **APSA**
