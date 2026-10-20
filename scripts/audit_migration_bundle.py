#!/usr/bin/env python3
"""Stream-audit a persistent-data tar archive without unpacking it to disk.

Reads a plain tar stream on stdin (for example `zstd -dc persistent-data.tar.zst |`) and, in one
pass:

* verifies every regular member's SHA-256 against the pack-time manifest
  (`persistent-files.sha256`) and reports members missing from either side;
* scans every text-like member, including members inside nested `.tar.gz` and `.gz` files,
  for credential patterns (Hugging Face, GitHub, OpenAI, AWS, W&B, private keys, bearer
  tokens, generic token/password fields) and records redacted hits;
* extracts text-like members (reports, CSVs, logs, manifests) to OUT_DIR, skipping tensors
  and archives, up to a byte budget, so results can be inspected without the full archive.

Usage:
  zstd -dc persistent-data.tar.zst | audit_migration_bundle.py MANIFEST OUT_DIR REPORT.json
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time

TEXT_EXT = {
    ".json", ".jsonl", ".csv", ".tsv", ".log", ".txt", ".md", ".sha256", ".exit", ".pid",
    ".lock", ".yaml", ".yml", ".cfg", ".ini", ".toml", ".html", ".tex", ".bib", ".py", ".sh",
}
SKIP_EXT = {".pt", ".pth", ".npz", ".npy", ".safetensors", ".bin"}
MAX_TEXT_MEMBER = 1 << 30            # scan text members up to 1 GiB in memory
EXTRACT_BUDGET = int(os.environ.get("EXTRACT_BUDGET_BYTES", 12 * (1 << 30)))
CHUNK = 16 << 20

PATTERNS = [
    ("huggingface_token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("openai_key", re.compile(rb"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("aws_access_key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("private_key_block", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("bearer_token", re.compile(rb"Bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("wandb_key", re.compile(rb"(?i)wandb_api_key[\"'\s:=]+[0-9a-f]{40}")),
    ("slack_token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("google_api_key", re.compile(rb"\bAIza[0-9A-Za-z_\-]{35}\b")),
    (
        "credential_field",
        re.compile(
            rb"(?i)[\"']?(hf_token|huggingface_token|hugging_face_hub_token|api_key|apikey|"
            rb"secret_key|client_secret|access_token|auth_token|password|passwd)[\"']?\s*[:=]\s*"
            rb"[\"']?[A-Za-z0-9_\-\.]{12,}"
        ),
    ),
]


def ext_of(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    if base.endswith(".tar.gz"):
        return ".tar.gz"
    return os.path.splitext(base)[1].lower()


def redact(match: bytes) -> str:
    text = match.decode("utf-8", "replace")
    return text[:10] + "…" + f"[{len(text)} chars]"


def scan_bytes(name: str, data: bytes, hits: list, context: str) -> None:
    for label, pattern in PATTERNS:
        for m in pattern.finditer(data):
            hits.append({"member": name, "context": context, "pattern": label,
                         "offset": m.start(), "redacted": redact(m.group(0))})
            if len(hits) > 5000:
                return


class HashingReader(io.RawIOBase):
    """Wrap a sequential file object, hashing and counting everything read through it."""

    def __init__(self, fileobj):
        self.fileobj = fileobj
        self.sha = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def read(self, n=-1):
        data = self.fileobj.read(n)
        if data:
            self.sha.update(data)
            self.count += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def drain(self):
        while True:
            data = self.fileobj.read(CHUNK)
            if not data:
                break
            self.sha.update(data)
            self.count += len(data)


def scan_nested_targz(outer_name: str, reader, hits: list, stats: dict) -> None:
    try:
        gz = gzip.GzipFile(fileobj=reader)
        inner = tarfile.open(fileobj=gz, mode="r|")
        for m in inner:
            if not m.isfile():
                continue
            f = inner.extractfile(m)
            if f is None:
                continue
            e = ext_of(m.name)
            stats["nested_members"] += 1
            if e in SKIP_EXT or m.size > MAX_TEXT_MEMBER:
                # still must consume the stream
                while f.read(CHUNK):
                    pass
                continue
            data = f.read()
            if e in TEXT_EXT or e == "":
                stats["nested_text_scanned"] += 1
                scan_bytes(m.name, data, hits, f"inside {outer_name}")
    except Exception as error:  # noqa: BLE001 - record and continue
        stats.setdefault("nested_errors", []).append({"member": outer_name, "error": repr(error)})


def main() -> int:
    manifest_path, out_dir, report_path = sys.argv[1:4]
    manifest: dict[str, str] = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            digest, path = line.split("  ", 1)
            manifest[path] = digest
    os.makedirs(out_dir, exist_ok=True)

    hits: list = []
    stats = {
        "members": 0, "regular_files": 0, "bytes": 0, "sha_ok": 0, "sha_mismatch": [],
        "not_in_manifest": [], "extracted_files": 0, "extracted_bytes": 0, "extract_skipped_budget": 0,
        "text_scanned": 0, "nested_archives": 0, "nested_members": 0, "nested_text_scanned": 0,
        "largest_members": [],
    }
    seen: set[str] = set()
    started = time.time()
    tar = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
    for m in tar:
        stats["members"] += 1
        if not m.isfile():
            continue
        stats["regular_files"] += 1
        f = tar.extractfile(m)
        if f is None:
            continue
        name = m.name
        seen.add(name)
        e = ext_of(name)
        reader = HashingReader(f)
        if e == ".tar.gz":
            stats["nested_archives"] += 1
            scan_nested_targz(name, reader, hits, stats)
            reader.drain()
        elif e == ".gz" and m.size <= MAX_TEXT_MEMBER:
            # a plain gzip member (e.g. permutation_nulls.json.gz): scan the decompressed text
            # and keep the compressed file, which the summary scripts read directly
            raw = reader.read(m.size) if m.size else b""
            reader.drain()
            try:
                scan_bytes(name, gzip.decompress(raw), hits, "gzip member")
                stats["text_scanned"] += 1
            except Exception as error:  # noqa: BLE001
                stats.setdefault("gzip_errors", []).append({"member": name, "error": repr(error)})
            if stats["extracted_bytes"] + len(raw) <= EXTRACT_BUDGET:
                target = os.path.join(out_dir, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as out:
                    out.write(raw)
                stats["extracted_files"] += 1
                stats["extracted_bytes"] += len(raw)
            else:
                stats["extract_skipped_budget"] += 1
        elif e == ".gz":
            reader.drain()
        elif e in SKIP_EXT or m.size > MAX_TEXT_MEMBER:
            reader.drain()
        else:
            data = reader.read(m.size) if m.size else b""
            reader.drain()
            stats["text_scanned"] += 1
            scan_bytes(name, data, hits, "member")
            if stats["extracted_bytes"] + len(data) <= EXTRACT_BUDGET:
                target = os.path.join(out_dir, name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as out:
                    out.write(data)
                stats["extracted_files"] += 1
                stats["extracted_bytes"] += len(data)
            else:
                stats["extract_skipped_budget"] += 1
        digest = reader.sha.hexdigest()
        stats["bytes"] += reader.count
        expected = manifest.get(name)
        if expected is None:
            stats["not_in_manifest"].append(name)
        elif expected == digest:
            stats["sha_ok"] += 1
        else:
            stats["sha_mismatch"].append({"member": name, "expected": expected, "actual": digest})
        stats["largest_members"].append((m.size, name))
        stats["largest_members"] = sorted(stats["largest_members"], reverse=True)[:25]
        if stats["regular_files"] % 100 == 0:
            print(f"[{time.time() - started:7.0f}s] {stats['regular_files']} files, "
                  f"{stats['bytes'] / 1e9:.1f} GB, hits={len(hits)}", file=sys.stderr, flush=True)

    stats["manifest_entries"] = len(manifest)
    stats["missing_from_archive"] = sorted(set(manifest) - seen)
    stats["elapsed_seconds"] = round(time.time() - started, 1)
    stats["credential_hits"] = hits
    stats["largest_members"] = [{"bytes": b, "member": n} for b, n in stats["largest_members"]]
    stats["verdict"] = (
        "PASS" if not stats["sha_mismatch"] and not stats["missing_from_archive"]
        and not stats["not_in_manifest"] else "FAIL"
    )
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=1)
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("credential_hits", "largest_members", "missing_from_archive")},
                     indent=1))
    print(f"credential hits: {len(hits)}; missing from archive: {len(stats['missing_from_archive'])}")
    return 0 if stats["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
