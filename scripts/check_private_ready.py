#!/usr/bin/env python3
"""Fail if prospective meta-repository files are unsafe or non-portable."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 95 * 1024 * 1024
EXCLUDED_ROOTS = ("logs", "wandb", "transformers", "transformers_qwenvl", "vllm", "vLLM-Hook", "IVC-Prune")
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(rb"\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b"),
    "OpenAI-style key": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def git_output(*args: str) -> bytes:
    return subprocess.run(("git", *args), cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []

    try:
        git_output("rev-parse", "--show-toplevel")
    except subprocess.CalledProcessError as error:
        raise SystemExit("Workspace root is not initialized as a Git repository") from error

    for name in EXCLUDED_ROOTS:
        # A trailing slash asks Git to test the directory pattern even when the
        # optional checkout/output directory does not exist on this machine.
        ignored = subprocess.run(("git", "check-ignore", "--quiet", "--no-index", f"{name}/"), cwd=ROOT).returncode == 0
        if not ignored:
            failures.append(f"not ignored: {name}/")

    raw_paths = git_output("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    candidates = [ROOT / item.decode("utf-8", errors="surrogateescape") for item in raw_paths.split(b"\0") if item]
    site_path_marker = b"/" + b"idiap/"
    site_email_marker = b"@" + b"idiap.ch"
    for path in candidates:
        if not path.is_file():
            continue
        size = path.stat().st_size
        relative = path.relative_to(ROOT)
        if size > MAX_FILE_BYTES:
            failures.append(f"oversized candidate ({size} bytes): {relative}")
            continue
        data = path.read_bytes()
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(data):
                failures.append(f"{label} pattern: {relative}")
        if site_path_marker in data or site_email_marker in data:
            warnings.append(f"site-specific path or identity: {relative}")

    lock = json.loads((ROOT / "repos.lock.json").read_text(encoding="utf-8"))
    overlay_specs = list(lock["repositories"].values()) + list(lock["archived_refs"].values())
    for spec in overlay_specs:
        overlay_name = spec.get("overlay")
        if not overlay_name:
            continue
        overlay = ROOT / overlay_name
        if not overlay.is_file():
            failures.append(f"missing overlay: {overlay_name}")
            continue
        expected = spec["overlay_sha256"]
        actual = file_sha256(overlay)
        if expected == "PENDING" or expected != actual:
            failures.append(f"checksum mismatch: {overlay_name}")
        if spec.get("overlay_size_bytes") != overlay.stat().st_size:
            failures.append(f"size mismatch: {overlay_name}")

    for warning in sorted(set(warnings)):
        print(f"WARNING: {warning}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        raise SystemExit(1)
    print(f"private-ready checks passed for {len(candidates)} candidate files")


if __name__ == "__main__":
    main()
