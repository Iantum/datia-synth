# Changelog

All notable changes to datia-synth will be documented in this file.

## [1.0.0] — 2026-03-30

### Added
- `DatiaSynthesizer` — GaussianCopulaSynthesizer (SDV) with adaptive binary search
- `DataQualityValidator` — 9-metric quality scoring:
  - Statistical: Wasserstein, KL-divergence, KS-test
  - Geometric: PCA centroid distance, UMAP Local Neighborhood Preservation
  - Topological: TDA persistence diagrams H0+H1 (ripser + persim)
  - Semantic: domain rule validation, intersectional profile consistency
- `PostQuantumSigner` — ML-DSA-44 (FIPS 204) batch signing and verification
- `SphincsBackupSigner` — SLH-DSA-SHA2-128s (FIPS 205) backup signing
- `KyberKeyEncapsulator` — ML-KEM-512 (FIPS 203) key encapsulation
- FastAPI REST API with 9 endpoints including `/visualize` (UMAP/t-SNE/PCA projection)
- Graceful fallback when optional dependencies (umap-learn, ripser, liboqs) are unavailable
- Dockerfile for containerised deployment
- Apache 2.0 licence
