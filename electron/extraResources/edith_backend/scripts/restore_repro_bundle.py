#!/usr/bin/env python3
"""
Restore Edith reproducibility bundle created by export_repro_bundle.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Restore Edith reproducibility bundle")
    p.add_argument("bundle", help="Path to .tar.gz bundle")
    p.add_argument("--project-root", default="", help="Override project root")
    p.add_argument("--app-data-dir", default="", help="Override app state directory")
    p.add_argument("--restore-secrets", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--restore-chat-history", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--restore-chroma", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--restore-snapshots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--restore-portability-assets", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env():
    if not load_dotenv:
        return
    root = project_root()
    env = root / ".env"
    if env.exists():
        load_dotenv(dotenv_path=env, override=False)


def app_data_dir() -> Path:
    raw = (os.getenv("EDITH_APP_DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "Library" / "Application Support" / "Edith").resolve()


def copy_file(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path, force: bool):
    if not src.exists():
        return False
    if dst.exists():
        if not force:
            raise RuntimeError(f"Destination exists (use --force): {dst}")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return True


def safe_extract_tar(tar: tarfile.TarFile, target: Path):
    root = target.resolve()
    members = tar.getmembers()
    for member in members:
        resolved = (root / member.name).resolve()
        if not str(resolved).startswith(str(root)):
            raise RuntimeError(f"Blocked unsafe archive path: {member.name}")
    kwargs = {}
    if "filter" in tarfile.TarFile.extractall.__code__.co_varnames:
        kwargs["filter"] = "data"
    tar.extractall(root, **kwargs)


def main() -> int:
    load_env()
    args = parse_args()
    bundle = Path(args.bundle).expanduser().resolve()
    if not bundle.exists():
        raise SystemExit(f"Bundle not found: {bundle}")

    root = Path(args.project_root).expanduser().resolve() if args.project_root else project_root()
    app_data = Path(args.app_data_dir).expanduser().resolve() if args.app_data_dir else app_data_dir()

    with tempfile.TemporaryDirectory(prefix="edith_restore_") as tmp:
        tmp_dir = Path(tmp)
        with tarfile.open(bundle, "r:gz") as tar:
            safe_extract_tar(tar, tmp_dir)

        stage = tmp_dir / "edith_repro_bundle"
        if not stage.exists():
            raise SystemExit("Invalid bundle format: edith_repro_bundle directory missing.")

        restored = {}

        restored["eval_cases"] = copy_file(stage / "project" / "eval" / "cases.jsonl", root / "eval" / "cases.jsonl")
        restored["eval_traps"] = copy_file(
            stage / "project" / "eval" / "hallucination_traps.jsonl",
            root / "eval" / "hallucination_traps.jsonl",
        )

        if args.restore_secrets:
            restored["project_env"] = copy_file(stage / "project" / ".env", root / ".env")
        else:
            restored["project_env"] = False

        restored["retrieval_profile"] = copy_file(stage / "app_state" / "retrieval_profile.json", app_data / "retrieval_profile.json")
        restored["index_report"] = copy_file(stage / "app_state" / "edith_index_report.csv", app_data / "edith_index_report.csv")
        restored["feedback_db"] = copy_file(stage / "app_state" / "feedback.sqlite3", app_data / "feedback.sqlite3")
        restored["run_ledger"] = copy_file(stage / "app_state" / "run_ledger.jsonl", app_data / "run_ledger.jsonl")
        restored["run_ledger_encrypted"] = copy_file(
            stage / "app_state" / "run_ledger.jsonl.enc",
            app_data / "run_ledger.jsonl.enc",
        )
        restored["desktop_config"] = copy_file(stage / "app_state" / "config.json", app_data / "config.json")
        restored["glossary_graph"] = copy_file(stage / "app_state" / "glossary_graph.json", app_data / "glossary_graph.json")
        restored["citation_graph"] = copy_file(stage / "app_state" / "citation_graph.json", app_data / "citation_graph.json")
        restored["chapter_anchors"] = copy_file(stage / "app_state" / "chapter_anchors.json", app_data / "chapter_anchors.json")
        restored["claim_inventory"] = copy_file(stage / "app_state" / "claim_inventory.json", app_data / "claim_inventory.json")
        restored["experiment_ledger"] = copy_file(stage / "app_state" / "experiment_ledger.json", app_data / "experiment_ledger.json")
        restored["bibliography_db"] = copy_file(stage / "app_state" / "bibliography_db.json", app_data / "bibliography_db.json")
        restored["entity_timeline"] = copy_file(stage / "app_state" / "entity_timeline.json", app_data / "entity_timeline.json")
        restored["index_health_report"] = copy_file(stage / "app_state" / "index_health_report.json", app_data / "index_health_report.json")

        if args.restore_chat_history:
            restored["chat_history"] = copy_tree(
                stage / "app_state" / "chat_history",
                app_data / "chat_history",
                force=bool(args.force),
            )
        else:
            restored["chat_history"] = False

        if args.restore_chroma:
            restored["chroma"] = copy_tree(
                stage / "app_state" / "chroma",
                app_data / "chroma",
                force=bool(args.force),
            )
        else:
            restored["chroma"] = False

        if args.restore_snapshots:
            restored["snapshots"] = copy_tree(
                stage / "app_state" / "snapshots",
                app_data / "snapshots",
                force=bool(args.force),
            )
        else:
            restored["snapshots"] = False

        if args.restore_portability_assets:
            restored["templates_dir"] = copy_tree(
                stage / "project" / "templates",
                root / "templates",
                force=bool(args.force),
            )
            restored["ontology_dir"] = copy_tree(
                stage / "project" / "ontology",
                root / "ontology",
                force=bool(args.force),
            )
            restored["skills_dir"] = copy_tree(
                stage / "project" / "skills",
                root / "skills",
                force=bool(args.force),
            )
        else:
            restored["templates_dir"] = False
            restored["ontology_dir"] = False
            restored["skills_dir"] = False

    print(
        json.dumps(
            {
                "ok": True,
                "bundle": str(bundle),
                "project_root": str(root),
                "app_data_dir": str(app_data),
                "restore_secrets": bool(args.restore_secrets),
                "restore_chat_history": bool(args.restore_chat_history),
                "restore_chroma": bool(args.restore_chroma),
                "restore_snapshots": bool(args.restore_snapshots),
                "restore_portability_assets": bool(args.restore_portability_assets),
                "restored": restored,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
