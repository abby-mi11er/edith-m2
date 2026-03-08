"""
TLS configuration for Edith server.

Provides helpers to run uvicorn with HTTPS when certificates are available.
For local dev, falls back to HTTP on 127.0.0.1.

Usage:
    python -m server.tls_config   # prints current TLS status
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path
from typing import Optional


def get_tls_config() -> dict:
    """Return uvicorn SSL kwargs if certs are configured, else empty dict.

    Environment variables:
        EDITH_TLS_CERT:  Path to PEM certificate file
        EDITH_TLS_KEY:   Path to PEM private key file
        EDITH_TLS_CA:    (Optional) Path to CA bundle for client verification
    """
    cert = os.environ.get("EDITH_TLS_CERT", "")
    key = os.environ.get("EDITH_TLS_KEY", "")

    if not cert or not key:
        return {}

    cert_path = Path(cert)
    key_path = Path(key)

    if not cert_path.exists():
        print(f"  [tls] WARNING: cert file not found: {cert}")
        return {}
    if not key_path.exists():
        print(f"  [tls] WARNING: key file not found: {key}")
        return {}

    config = {
        "ssl_certfile": str(cert_path),
        "ssl_keyfile": str(key_path),
    }

    # Optional CA bundle for mutual TLS
    ca = os.environ.get("EDITH_TLS_CA", "")
    if ca and Path(ca).exists():
        config["ssl_ca_certs"] = ca
        config["ssl_cert_reqs"] = ssl.CERT_OPTIONAL

    return config


def is_tls_enabled() -> bool:
    """Check if TLS is configured."""
    return bool(get_tls_config())


def generate_self_signed_cert(
    cert_dir: str = "",
    hostname: str = "localhost",
) -> tuple[str, str]:
    """Generate a self-signed certificate for local development.

    Returns (cert_path, key_path).
    Requires the `cryptography` package (already in requirements.txt).
    """
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    if not cert_dir:
        cert_dir = str(Path(__file__).parent.parent / "certs")
    os.makedirs(cert_dir, exist_ok=True)

    cert_path = os.path.join(cert_dir, "edith_dev.pem")
    key_path = os.path.join(cert_dir, "edith_dev_key.pem")

    # Generate key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Generate certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Edith Dev"),
    ])

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(ipaddress_from_str("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # Write key
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    os.chmod(key_path, 0o600)

    # Write cert
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path


def ipaddress_from_str(addr: str):
    """Convert string to ipaddress object."""
    import ipaddress
    return ipaddress.IPv4Address(addr)


if __name__ == "__main__":
    if is_tls_enabled():
        config = get_tls_config()
        print(f"TLS enabled: cert={config.get('ssl_certfile')}")
    else:
        print("TLS not configured. Set EDITH_TLS_CERT and EDITH_TLS_KEY.")
        print("To generate a self-signed dev cert:")
        print("  python -c \"from server.tls_config import generate_self_signed_cert; print(generate_self_signed_cert())\"")
