#!/usr/bin/env python3
"""
Run repeatable dependency and lockfile security checks for Edith.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path, timeout_s: int = 300) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s)
        return proc.returncode, (proc.stdout or ""), (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "", f"Command timed out after {timeout_s}s: {' '.join(cmd)}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Edith security checks")
    p.add_argument("--ci", action="store_true", help="Fail on warnings in CI mode")
    p.add_argument("--json-out", default="", help="Optional JSON report output path")
    return p.parse_args()


def lockfile_check(root: Path) -> dict:
    issues: list[str] = []
    req = root / "requirements.txt"
    pkg = root / "electron" / "package.json"
    lock = root / "electron" / "package-lock.json"

    if not req.exists():
        issues.append("Missing requirements.txt")
    if not pkg.exists():
        issues.append("Missing electron/package.json")
    if not lock.exists():
        issues.append("Missing electron/package-lock.json")
    else:
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            if int(payload.get("lockfileVersion", 0)) < 2:
                issues.append("electron/package-lock.json lockfileVersion must be >= 2")
        except Exception as e:
            issues.append(f"Invalid package-lock.json: {e}")

    return {
        "name": "lockfiles",
        "status": "pass" if not issues else "fail",
        "details": "Lockfile policy satisfied." if not issues else "; ".join(issues),
    }


def pip_check(root: Path) -> dict:
    code, out, err = run_cmd([sys.executable, "-m", "pip", "check"], root)
    detail = (out or err).strip()
    return {
        "name": "pip_check",
        "status": "pass" if code == 0 else "fail",
        "details": detail,
    }


def pip_audit_check(root: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "pip_audit",
        "-r",
        "requirements.txt",
        "--format",
        "json",
        "--progress-spinner",
        "off",
    ]
    code, out, err = run_cmd(cmd, root)
    text = (out or err).strip()

    if code != 0 and ("No module named" in text or "No module named pip_audit" in text):
        venv_py = root / ".venv" / "bin" / "python"
        if venv_py.exists():
            code, out, err = run_cmd(
                [
                    str(venv_py),
                    "-m",
                    "pip_audit",
                    "-r",
                    "requirements.txt",
                    "--format",
                    "json",
                    "--progress-spinner",
                    "off",
                ],
                root,
            )
            text = (out or err).strip()
        else:
            return {
                "name": "pip_audit",
                "status": "warn",
                "details": "pip-audit is not installed. Install with: python -m pip install pip-audit",
            }

    try:
        payload = json.loads(out or "{}")
    except Exception:
        payload = {}

    vuln_count = 0
    deps = payload.get("dependencies") if isinstance(payload, dict) else None
    if isinstance(deps, list):
        for dep in deps:
            vulns = dep.get("vulns") if isinstance(dep, dict) else []
            vuln_count += len(vulns or [])

    if code == 0 and vuln_count == 0:
        return {
            "name": "pip_audit",
            "status": "pass",
            "details": "No known Python package vulnerabilities.",
        }

    if vuln_count > 0:
        return {
            "name": "pip_audit",
            "status": "fail",
            "details": f"Detected {vuln_count} Python package vulnerabilities.",
        }

    if "Failed to upgrade `pip`" in text:
        return {
            "name": "pip_audit",
            "status": "warn",
            "details": (
                "pip-audit could not create its temporary resolver environment in this runtime. "
                "Run from the project .venv on a machine with full pip network access."
            ),
        }

    return {
        "name": "pip_audit",
        "status": "warn",
        "details": text or "pip-audit returned a non-zero status without parsable JSON.",
    }


def npm_audit_check(root: Path) -> dict:
    electron_dir = root / "electron"
    cmd = ["npm", "audit", "--audit-level=high", "--json"]
    code, out, err = run_cmd(cmd, electron_dir)
    text = (out or err).strip()

    if "ENOTFOUND" in text or "EAI_AGAIN" in text:
        return {
            "name": "npm_audit",
            "status": "warn",
            "details": "npm audit could not reach registry (network/DNS issue).",
        }

    try:
        payload = json.loads(out or "{}")
    except Exception:
        payload = {}

    vul = (payload.get("metadata") or {}).get("vulnerabilities") if isinstance(payload, dict) else None
    high = int((vul or {}).get("high", 0))
    critical = int((vul or {}).get("critical", 0))
    total = high + critical

    if total == 0 and code == 0:
        return {
            "name": "npm_audit",
            "status": "pass",
            "details": "No high/critical npm vulnerabilities.",
        }

    if total > 0:
        return {
            "name": "npm_audit",
            "status": "fail",
            "details": f"Detected {high} high and {critical} critical npm vulnerabilities.",
        }

    return {
        "name": "npm_audit",
        "status": "warn",
        "details": text or "npm audit returned a non-zero status.",
    }


def app_security_posture_check(root: Path) -> dict:
    issues: list[str] = []
    main_js = root / "electron" / "main.js"
    launcher = root / "desktop_launcher.py"
    app_py = root / "app.py"
    try:
        text = main_js.read_text(encoding="utf-8")
        for required in (
            "contextIsolation: true",
            "nodeIntegration: false",
            "sandbox: true",
            "webSecurity: true",
        ):
            if required not in text:
                issues.append(f"electron/main.js missing `{required}`")
    except Exception as e:
        issues.append(f"Could not read electron/main.js: {e}")

    try:
        text = launcher.read_text(encoding="utf-8")
        for required in (
            "\"server.enableCORS\": True",
            "\"server.enableXsrfProtection\": True",
        ):
            if required not in text:
                issues.append(f"desktop_launcher.py missing `{required}`")
    except Exception as e:
        issues.append(f"Could not read desktop_launcher.py: {e}")

    try:
        text = app_py.read_text(encoding="utf-8")
        for required in (
            "REQUIRE_HTTPS_WEB_SOURCES",
            "EDITH_RATE_LIMIT_ENABLED",
            "EDITH_OAUTH_REQUIRED",
            "EDITH_RBAC_DEFAULT_ROLE",
        ):
            if required not in text:
                issues.append(f"app.py missing `{required}`")
    except Exception as e:
        issues.append(f"Could not read app.py: {e}")

    return {
        "name": "app_security_posture",
        "status": "pass" if not issues else "fail",
        "details": "Security defaults present." if not issues else "; ".join(issues),
    }


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    checks = [
        lockfile_check(root),
        app_security_posture_check(root),
        pip_check(root),
        pip_audit_check(root),
        npm_audit_check(root),
    ]

    fail = [c for c in checks if c["status"] == "fail"]
    warn = [c for c in checks if c["status"] == "warn"]

    report = {
        "ok": not fail and (not args.ci or not warn),
        "checks": checks,
        "fail_count": len(fail),
        "warn_count": len(warn),
    }

    if args.json_out:
        out_path = Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
