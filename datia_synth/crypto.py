"""
datia_synth.crypto
------------------
Criptografía post-cuántica para integridad del dato sintético (T1.2 — DatIA).

Estándares implementados:
  - FIPS 204 — ML-DSA-44 (CRYSTALS-Dilithium2) — firma digital principal
  - FIPS 205 — SLH-DSA-SHA2-128s (SPHINCS+) — firma de respaldo (sin estado)
  - FIPS 203 — ML-KEM-512 (Kyber) — cifrado de clave para transporte seguro

Todos los métodos tienen fallback gracioso: si liboqs no está disponible o
el algoritmo concreto no está compilado en la librería, devuelven un valor
descriptivo en lugar de elevar una excepción, para no bloquear la generación.
"""

from __future__ import annotations

import json
from typing import Tuple


# ---------------------------------------------------------------------------
#  FIPS 204 — ML-DSA-44 (CRYSTALS-Dilithium2)
# ---------------------------------------------------------------------------

class PostQuantumSigner:
    """
    Firmante post-cuántico basado en ML-DSA-44 (CRYSTALS-Dilithium2 / FIPS 204).

    Requiere liboqs-python >= 0.10.0

    Uso::

        signer = PostQuantumSigner()
        sig_hex, pub_hex = signer.sign(records)
        ok = signer.verify(records, sig_hex, pub_hex)
    """

    ALGORITHM = "ML-DSA-44"

    def sign(self, data: list) -> Tuple[str, str]:
        """
        Firma una lista de registros con ML-DSA-44.

        Retorna (signature_hex, public_key_hex).
        Si liboqs no está disponible o el algoritmo no está habilitado,
        devuelve un par de strings de error (no eleva excepción) para
        que el servicio pueda continuar.
        """
        try:
            import oqs
            enabled = oqs.get_enabled_sig_mechanisms()
            if self.ALGORITHM not in enabled:
                return "ALGORITMO_PENDIENTE_CONFIG", "KEY_PENDIENTE"

            with oqs.Signature(self.ALGORITHM) as s:
                pub_key = s.generate_keypair()
                payload = json.dumps(data, sort_keys=True).encode("utf-8")
                signature = s.sign(payload)
                return signature.hex(), pub_key.hex()
        except Exception as exc:
            return f"ERROR_MOTOR: {exc}", "REVISAR_LIBOQS_V_COMPAT"

    def verify(self, data: list, signature_hex: str, public_key_hex: str) -> bool:
        """Verifica la firma ML-DSA-44 de una lista de registros."""
        try:
            import oqs
            with oqs.Signature(self.ALGORITHM) as v:
                payload = json.dumps(data, sort_keys=True).encode("utf-8")
                return v.verify(
                    payload,
                    bytes.fromhex(signature_hex),
                    bytes.fromhex(public_key_hex),
                )
        except Exception as exc:
            import logging
            logging.getLogger("datia.crypto").warning("ML-DSA-44 verify failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
#  FIPS 205 — SLH-DSA-SHA2-128s (SPHINCS+) — firma de respaldo sin estado
# ---------------------------------------------------------------------------

class SphincsBackupSigner:
    """
    Firma de respaldo post-cuántica SLH-DSA-SHA2-128s (SPHINCS+ / FIPS 205).

    Stateless hash-based signature — no requiere gestión de estado de claves.
    Usar como firma de respaldo cuando se requiere doble firma o rotación de claves.

    Uso::

        backup = SphincsBackupSigner()
        sig_hex, pub_hex = backup.sign(records)
        ok = backup.verify(records, sig_hex, pub_hex)
    """

    # SPHINCS+ variante compacta: balance velocidad/seguridad nivel 1
    ALGORITHM = "SPHINCS+-SHA2-128s-simple"

    def sign(self, data: list) -> Tuple[str, str]:
        """Firma con SLH-DSA (SPHINCS+). Retorna (signature_hex, public_key_hex)."""
        try:
            import oqs
            enabled = oqs.get_enabled_sig_mechanisms()
            if self.ALGORITHM not in enabled:
                return "SLH_DSA_PENDIENTE_CONFIG", "KEY_PENDIENTE"
            with oqs.Signature(self.ALGORITHM) as s:
                pub_key = s.generate_keypair()
                payload = json.dumps(data, sort_keys=True).encode("utf-8")
                signature = s.sign(payload)
                return signature.hex(), pub_key.hex()
        except Exception as exc:
            return f"SLH_DSA_ERROR: {exc}", "REVISAR_LIBOQS"

    def verify(self, data: list, signature_hex: str, public_key_hex: str) -> bool:
        """Verifica la firma SLH-DSA de una lista de registros."""
        try:
            import oqs
            with oqs.Signature(self.ALGORITHM) as v:
                payload = json.dumps(data, sort_keys=True).encode("utf-8")
                return v.verify(
                    payload,
                    bytes.fromhex(signature_hex),
                    bytes.fromhex(public_key_hex),
                )
        except Exception as exc:
            import logging
            logging.getLogger("datia.crypto").warning("SLH-DSA verify failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
#  FIPS 203 — ML-KEM-512 (Kyber) — intercambio de clave post-cuántico
# ---------------------------------------------------------------------------

class KyberKeyEncapsulator:
    """
    Intercambio de clave post-cuántico ML-KEM-512 (Kyber / FIPS 203).

    Permite cifrar una clave simétrica (p.ej. AES-256) para transporte seguro
    entre el servidor de generación (api_datia) y el receptor de los datos.

    Protocolo estándar KEM:
      Sender:
        pk, sk = encapsulator.generate_keypair()  → publicar pk
      Receiver:
        ciphertext, shared_secret = encapsulator.encapsulate(pk)  → enviar ciphertext
      Sender:
        shared_secret = encapsulator.decapsulate(ciphertext, sk)

    Uso::

        kem = KyberKeyEncapsulator()
        pk_hex, sk_hex = kem.generate_keypair()
        ct_hex, ss_hex = kem.encapsulate(pk_hex)
        ss_recovered = kem.decapsulate(ct_hex, sk_hex)
        assert ss_hex == ss_recovered
    """

    ALGORITHM = "ML-KEM-512"

    def generate_keypair(self) -> Tuple[str, str]:
        """Genera par de claves KEM. Retorna (public_key_hex, secret_key_hex)."""
        try:
            import oqs
            enabled = oqs.get_enabled_kem_mechanisms()
            if self.ALGORITHM not in enabled:
                return "ML_KEM_PENDIENTE_CONFIG", "SK_PENDIENTE"
            with oqs.KeyEncapsulation(self.ALGORITHM) as kem:
                pub_key = kem.generate_keypair()
                secret_key = kem.export_secret_key() if hasattr(kem, 'export_secret_key') else kem.secret_key
                if isinstance(secret_key, memoryview):
                    secret_key = bytes(secret_key)
                return pub_key.hex(), secret_key.hex()
        except Exception as exc:
            return f"ML_KEM_ERROR: {exc}", "REVISAR_LIBOQS"

    def encapsulate(self, public_key_hex: str) -> Tuple[str, str]:
        """
        Encapsula una clave simétrica usando la clave pública del receptor.
        Retorna (ciphertext_hex, shared_secret_hex).
        """
        try:
            import oqs
            with oqs.KeyEncapsulation(self.ALGORITHM) as kem:
                pub_key = bytes.fromhex(public_key_hex)
                ciphertext, shared_secret = kem.encap_secret(pub_key)
                return ciphertext.hex(), shared_secret.hex()
        except Exception as exc:
            return f"ENCAP_ERROR: {exc}", "ERROR"

    def decapsulate(self, ciphertext_hex: str, secret_key_hex: str) -> str:
        """
        Decapsula el ciphertext con la clave secreta para recuperar el shared secret.
        Retorna shared_secret_hex o string de error.
        """
        try:
            import oqs
            with oqs.KeyEncapsulation(self.ALGORITHM, secret_key=bytes.fromhex(secret_key_hex)) as kem:
                shared_secret = kem.decap_secret(bytes.fromhex(ciphertext_hex))
                return shared_secret.hex()
        except Exception as exc:
            return f"DECAP_ERROR: {exc}"
