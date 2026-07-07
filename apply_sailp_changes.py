#!/usr/bin/env python3
"""Apply the SAI-LP overlay to a local mfe-ascend fork/clone.

Usage:
    cd /path/to/your/mfe-ascend
    python /path/to/sailp_overlay/apply_sailp_changes.py

The script copies modified/new files into the repository root and backs up any
existing files under .sailp_backup/<timestamp>/ before overwriting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

FILES = [
    "mfe/optimizers/sailp.py",
    "mfe/optimizers/multi_request.py",
    "mfe/optimizers/__init__.py",
    "mfe/parser.py",
    "mfe/components/operator.py",
    "mfe/components/model_config.py",
    "mfe/components/query.py",
    "mfe/components/__init__.py",
    "mfe/workers/worker_v.py",
    "templates/sailp_example.yaml",
    "docs/sailp.md",
    "README_SAILP.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply SAI-LP overlay to mfe-ascend")
    parser.add_argument("--repo", default=".", help="Path to the mfe-ascend repository root")
    parser.add_argument("--no-backup", action="store_true", help="Do not create backups before overwriting files")
    args = parser.parse_args()

    src_root = Path(__file__).resolve().parent
    repo = Path(args.repo).resolve()
    if not (repo / "mfe").is_dir() or not (repo / "pyproject.toml").exists():
        print(f"ERROR: {repo} does not look like an mfe-ascend repository root", file=sys.stderr)
        return 2

    backup_root = repo / ".sailp_backup" / time.strftime("%Y%m%d-%H%M%S")
    copied = []
    for rel in FILES:
        src = src_root / rel
        dst = repo / rel
        if not src.exists():
            print(f"ERROR: overlay file missing: {src}", file=sys.stderr)
            return 2
        if dst.exists() and not args.no_backup:
            backup = backup_root / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    print(f"Applied SAI-LP overlay to {repo}")
    if not args.no_backup:
        print(f"Backups saved under {backup_root}")
    print("Copied files:")
    for rel in copied:
        print(f"  - {rel}")
    print("\nNext steps:")
    print("  python -m py_compile mfe/optimizers/sailp.py mfe/optimizers/multi_request.py mfe/parser.py")
    print("  export MFE_SCHEDULER=sailp")
    print("  export MFE_ENABLE_PREFIX_CACHING=1  # optional, requires vLLM support")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
