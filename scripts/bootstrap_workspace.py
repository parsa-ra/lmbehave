#!/usr/bin/env python3
"""Recreate the pinned child repositories and apply the research overlays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional


META_ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, cwd: Optional[Path] = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_worktree(repo: Path, base: str) -> bytes:
    """Return the same base-to-worktree patch stream as snapshot_overlays.sh."""

    diff_options = ("--binary", "--full-index", "--no-ext-diff", "--no-textconv", "--no-renames")
    tracked = subprocess.run(
        ("git", "diff", *diff_options, base, "--"),
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked = subprocess.run(
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    chunks = [tracked]
    for raw_path in untracked.split(b"\0"):
        if not raw_path:
            continue
        relative = os.fsdecode(raw_path)
        result = subprocess.run(
            ("git", "diff", *diff_options, "--no-index", "--", "/dev/null", relative),
            cwd=repo,
            stdout=subprocess.PIPE,
        )
        if result.returncode != 1:
            raise RuntimeError(f"Could not snapshot untracked path: {repo / relative}")
        chunks.append(result.stdout)
    return b"".join(chunks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        default=META_ROOT,
        help="Meta-repository root to populate (default: the checkout containing this script)",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Clone from existing local child repositories instead of network origins",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also clone vllm, vLLM-Hook, and IVC-Prune",
    )
    return parser.parse_args()


def verify_existing_repository(name: str, target: Path, spec: dict) -> None:
    """Verify an existing ignored checkout instead of overwriting or silently skipping it."""

    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=target,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"Existing path is not a Git worktree: {target}")

    base = spec["checkout"]
    run("git", "cat-file", "-e", f"{base}^{{commit}}", cwd=target)
    overlay_name = spec.get("overlay")
    if overlay_name:
        overlay = META_ROOT / overlay_name
        expected = spec["overlay_sha256"]
        actual = sha256(overlay)
        if expected == "PENDING" or actual != expected:
            raise SystemExit(f"Overlay checksum mismatch: {overlay}")
        if snapshot_worktree(target, base) != overlay.read_bytes():
            raise SystemExit(f"Existing repository does not match its pinned overlay: {target}")
    else:
        actual_head = result.stdout.strip()
        if actual_head != base:
            raise SystemExit(f"Existing repository {name} is at {actual_head}, expected {base}")
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=normal"),
            cwd=target,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        if status:
            raise SystemExit(f"Existing pinned repository is dirty: {target}")

    run("git", "diff", "--check", cwd=target)
    print(f"verified existing {name} against {base}")


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    lock = json.loads((META_ROOT / "repos.lock.json").read_text(encoding="utf-8"))
    for name, spec in lock["repositories"].items():
        if not spec["required"] and not args.include_optional:
            continue

        target = destination / name
        if spec.get("submodule") and target.exists():
            result = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=target,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if result.returncode != 0:
                if not (destination / ".git").exists():
                    raise SystemExit(f"Uninitialized submodule outside a meta-repository checkout: {target}")
                run("git", "submodule", "update", "--init", "--checkout", "--", name, cwd=destination)
                result = subprocess.run(
                    ("git", "rev-parse", "HEAD"),
                    cwd=target,
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            actual = result.stdout.strip()
            if actual != spec["checkout"]:
                raise SystemExit(f"Submodule {name} is at {actual}, expected {spec['checkout']}")
            print(f"verified submodule {name} at {actual}")
            continue
        if target.exists():
            verify_existing_repository(name, target, spec)
            continue
        if args.source_root:
            source = (args.source_root.resolve() / name)
            if not (source / ".git").exists():
                raise SystemExit(f"Missing local source repository: {source}")
            run("git", "clone", "--local", "--no-hardlinks", "--no-checkout", str(source), str(target))
        else:
            run("git", "clone", "--no-checkout", spec["origin"], str(target))

        run("git", "-c", "core.autocrlf=false", "checkout", "--detach", spec["checkout"], cwd=target)

        overlay_name = spec.get("overlay")
        if overlay_name:
            overlay = META_ROOT / overlay_name
            expected = spec["overlay_sha256"]
            actual = sha256(overlay)
            if expected == "PENDING" or actual != expected:
                raise SystemExit(f"Overlay checksum mismatch: {overlay}")
            run("git", "apply", "--check", "--whitespace=nowarn", str(overlay), cwd=target)
            run("git", "apply", "--whitespace=nowarn", str(overlay), cwd=target)
            run("git", "apply", "--check", "--reverse", "--whitespace=nowarn", str(overlay), cwd=target)
            if snapshot_worktree(target, spec["checkout"]) != overlay.read_bytes():
                raise SystemExit(f"Replayed worktree does not exactly match overlay: {overlay}")

        run("git", "diff", "--check", cwd=target)
        print(f"prepared {name} at {spec['checkout']}")


if __name__ == "__main__":
    main()
