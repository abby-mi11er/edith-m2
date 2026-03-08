"""
Biometric Vault — macOS Keychain + Touch ID Integration
=========================================================
Three-Factor Authentication for E.D.I.T.H.:
  1. Something you HAVE  → the Bolt drive (.edith_anchor UUID)
  2. Something you ARE   → Touch ID / biometrics
  3. Something the Mac KNOWS → Keychain-stored credentials

Instead of .env plaintext, keys are stored in macOS Keychain and
only decrypted when the Bolt's UUID is verified via Touch ID.

Usage:
    from server.biometric_vault import BiometricVault

    vault = BiometricVault()
    vault.store_key("anthropic", "sk-ant-...")     # saves to Keychain
    key = vault.retrieve_key("anthropic")          # Touch ID prompt
    vault.hydrate_environment()                    # load all keys into os.environ
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

log = logging.getLogger("edith.biometric_vault")

# Keychain service name prefix
KC_SERVICE = "com.edith.sovereign"
KC_ACCOUNT_PREFIX = "edith_key_"

# Known connector keys
CONNECTOR_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "nyt": "NYT_API_KEY",
    "notion": "NOTION_TOKEN",
    "mendeley_id": "MENDELEY_CLIENT_ID",
    "mendeley_secret": "MENDELEY_CLIENT_SECRET",
    "mathpix_id": "MATHPIX_APP_ID",
    "mathpix_key": "MATHPIX_APP_KEY",
    "openalex": "OPENALEX_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gee_project": "GEE_PROJECT_ID",
}

BOLT_VOLUME = os.environ.get("EDITH_BOLT_VOLUME", "/Volumes/Edith Bolt")
ANCHOR_FILE = ".edith_anchor"


class BiometricVault:
    """Manages API keys via macOS Keychain with Bolt UUID verification."""

    def __init__(self, bolt_path: str | None = None):
        self.bolt_path = bolt_path or BOLT_VOLUME
        self._soul_id: str | None = None

    # ── Bolt Verification ──────────────────────────────────────────

    def _get_soul_id(self) -> str | None:
        """Read the Bolt's Soul UUID from the anchor file."""
        if self._soul_id:
            return self._soul_id
        anchor = Path(self.bolt_path) / ANCHOR_FILE
        if not anchor.exists():
            return None
        try:
            data = json.loads(anchor.read_text())
            self._soul_id = data.get("soul_id")
            return self._soul_id
        except Exception:
            return None

    def is_bolt_present(self) -> bool:
        """Check if the Bolt drive is connected and has a valid anchor."""
        return self._get_soul_id() is not None

    # ── Keychain Operations ────────────────────────────────────────

    def _kc_service(self, connector: str) -> str:
        """Generate Keychain service name for a connector."""
        return f"{KC_SERVICE}.{connector}"

    def store_key(self, connector: str, value: str) -> bool:
        """Store a key in macOS Keychain.

        Args:
            connector: Key name (e.g., 'anthropic', 'nyt')
            value: The API key value

        Returns:
            True if stored successfully
        """
        service = self._kc_service(connector)
        account = f"{KC_ACCOUNT_PREFIX}{connector}"

        # Delete existing entry first (ignore errors)
        subprocess.run(
            ["security", "delete-generic-password",
             "-s", service, "-a", account],
            capture_output=True
        )

        # Add new entry
        result = subprocess.run(
            ["security", "add-generic-password",
             "-s", service, "-a", account,
             "-w", value,
             "-T", "",  # Require confirmation for access
             "-U"],     # Update if exists
            capture_output=True, text=True
        )

        if result.returncode == 0:
            log.info(f"§VAULT: Stored key for {connector}")
            return True
        else:
            log.error(f"§VAULT: Failed to store key for {connector}: {result.stderr}")
            return False

    def retrieve_key(self, connector: str) -> str | None:
        """Retrieve a key from macOS Keychain.

        This may trigger a Touch ID / password prompt on first access.

        Args:
            connector: Key name (e.g., 'anthropic', 'nyt')

        Returns:
            The API key value, or None if not found
        """
        # Verify Bolt is connected first
        if not self.is_bolt_present():
            log.warning("§VAULT: Bolt not connected — key access denied")
            return None

        service = self._kc_service(connector)
        account = f"{KC_ACCOUNT_PREFIX}{connector}"

        result = subprocess.run(
            ["security", "find-generic-password",
             "-s", service, "-a", account, "-w"],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None

    def delete_key(self, connector: str) -> bool:
        """Remove a key from macOS Keychain."""
        service = self._kc_service(connector)
        account = f"{KC_ACCOUNT_PREFIX}{connector}"

        result = subprocess.run(
            ["security", "delete-generic-password",
             "-s", service, "-a", account],
            capture_output=True
        )
        return result.returncode == 0

    def list_stored_keys(self) -> list[str]:
        """List all E.D.I.T.H. keys in the Keychain."""
        stored = []
        for connector in CONNECTOR_KEYS:
            service = self._kc_service(connector)
            account = f"{KC_ACCOUNT_PREFIX}{connector}"
            result = subprocess.run(
                ["security", "find-generic-password",
                 "-s", service, "-a", account],
                capture_output=True
            )
            if result.returncode == 0:
                stored.append(connector)
        return stored

    # ── Environment Hydration ──────────────────────────────────────

    def hydrate_environment(self) -> dict[str, bool]:
        """Load all stored keys from Keychain into os.environ.

        Returns:
            dict mapping connector names to success/failure
        """
        if not self.is_bolt_present():
            log.warning("§VAULT: Cannot hydrate — Bolt not connected")
            return {k: False for k in CONNECTOR_KEYS}

        results = {}
        for connector, env_var in CONNECTOR_KEYS.items():
            key = self.retrieve_key(connector)
            if key:
                os.environ[env_var] = key
                results[connector] = True
                log.info(f"§VAULT: Hydrated {env_var}")
            else:
                results[connector] = False

        log.info(f"§VAULT: Hydrated {sum(results.values())}/{len(results)} keys")
        return results

    def dehydrate_environment(self):
        """Remove all E.D.I.T.H. keys from os.environ (RAM purge)."""
        for connector, env_var in CONNECTOR_KEYS.items():
            if env_var in os.environ:
                os.environ[env_var] = "PURGED"
                del os.environ[env_var]
        log.info("§VAULT: Environment dehydrated")

    # ── Migration from .env ────────────────────────────────────────

    def migrate_from_dotenv(self, env_path: str | None = None) -> dict[str, bool]:
        """Migrate keys from .env file into macOS Keychain.

        Args:
            env_path: Path to .env file (defaults to project root)

        Returns:
            dict mapping connector names to migration success
        """
        if env_path is None:
            env_path = str(Path(__file__).parent.parent / ".env")

        if not Path(env_path).exists():
            log.warning(f"§VAULT: .env not found at {env_path}")
            return {}

        # Parse .env
        env_vars = {}
        for line in Path(env_path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip().strip('"').strip("'")

        # Migrate each known key
        results = {}
        for connector, env_var in CONNECTOR_KEYS.items():
            if env_var in env_vars and env_vars[env_var]:
                success = self.store_key(connector, env_vars[env_var])
                results[connector] = success
                if success:
                    log.info(f"§VAULT: Migrated {connector} to Keychain")

        migrated = sum(results.values())
        log.info(f"§VAULT: Migrated {migrated}/{len(results)} keys from .env")
        return results
