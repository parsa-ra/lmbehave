#!/usr/bin/env python3
"""Verify persistent archives and restored representation artifacts by SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESTORED = (
    (ROOT / "logs" / "representation-dynamics-text-expansion-v1-v2", "completed_manifest.json", "artifacts"),
    (ROOT / "logs" / "apertus15-current-panel-v1", "completed_manifest.json", "artifacts"),
    (ROOT / "logs" / "musr-protocol-correction-v1", "correction_completion.json", "models"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restored_artifact_path(root: Path, original: str) -> Path:
    parts = Path(original).parts
    try:
        root_index = parts.index(root.name)
    except ValueError as error:
        raise ValueError(f"Original artifact path does not contain {root.name}: {original}") from error
    return root.joinpath(*parts[root_index + 1 :])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/home/ubuntu/data-imu-us-east3/vlmbehave-artifacts"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-archive-content-hashes",
        action="store_true",
        help="Audit checksum metadata and restored artifacts without rereading large archives.",
    )
    args = parser.parse_args()

    archive_rows: list[dict[str, Any]] = []
    for checksum_path in sorted(args.archive_root.glob("*/*.sha256")):
        fields = checksum_path.read_text(encoding="utf-8").strip().split()
        if len(fields) < 2 or len(fields[0]) != 64:
            raise ValueError(f"Malformed checksum file: {checksum_path}")
        archive_path = checksum_path.with_suffix("")
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)
        actual = None if args.skip_archive_content_hashes else sha256(archive_path)
        valid = None if actual is None else actual == fields[0]
        if valid is False:
            raise ValueError(f"Archive checksum mismatch: {archive_path}")
        archive_rows.append(
            {
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "expected_sha256": fields[0],
                "actual_sha256": actual,
                "content_hash_verified": valid,
            }
        )

    restored_rows: list[dict[str, Any]] = []
    for root, manifest_name, list_key in RESTORED:
        manifest_path = root / manifest_name
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = payload[list_key]
        for item in items:
            path = restored_artifact_path(root, str(item["artifact"]))
            expected = str(item["artifact_sha256"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            if actual != expected:
                raise ValueError(f"Restored artifact hash mismatch: {path}")
            restored_rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": actual,
                    "manifest": str(manifest_path),
                }
            )

    report = {
        "schema_version": 1,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "archive_root": str(args.archive_root),
        "archive_checksum_files": len(archive_rows),
        "archive_content_hashes_verified": not args.skip_archive_content_hashes,
        "archives": archive_rows,
        "restored_representation_artifacts": len(restored_rows),
        "restored_sets": {
            "current_model_text": 100,
            "apertus15_current_panel": 20,
            "musr_correction": 5,
        },
        "restored_artifacts": restored_rows,
    }
    output = args.output or (args.archive_root / "persistent_artifact_verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "archives": len(archive_rows),
                "restored_artifacts": len(restored_rows),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
