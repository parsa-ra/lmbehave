#!/usr/bin/env python3
"""Verify the frozen Apertus 1.0 pretraining refs without downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "apertus1_longitudinal_v1.json"


def request_bytes(url: str, token: str | None) -> bytes:
    headers = {"Accept": "application/json", "User-Agent": "vlmbehave-release-check/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def request_json(url: str, token: str | None) -> Any:
    return json.loads(request_bytes(url, token))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expected_stage(tokens_billion: int) -> str:
    if tokens_billion < 7000:
        return "stage1"
    if tokens_billion < 12000:
        return "stage3"
    if tokens_billion < 13500:
        return "stage4"
    return "stage5"


def evaluate(manifest: dict[str, Any], token: str | None) -> dict[str, Any]:
    checkpoints = manifest["checkpoints"]
    repositories = {checkpoint["repository"] for checkpoint in checkpoints}
    if len(repositories) != 1:
        raise ValueError("The Apertus 1.0 pretraining trajectory must use one repository")
    repository = repositories.pop()
    refs = request_json(f"https://huggingface.co/api/models/{repository}/refs", token)
    branches = {item["name"]: item["targetCommit"] for item in refs.get("branches", [])}

    rows = []
    config_hashes: set[str] = set()
    tokenizer_hashes: set[str] = set()
    for checkpoint in checkpoints:
        source_ref = checkpoint["source_ref"]
        expected_revision = checkpoint["revision"]
        observed_revision = branches.get(source_ref)
        stage_from_position = expected_stage(int(checkpoint["position"]))
        config_hash = None
        tokenizer_hash = None
        if observed_revision == expected_revision:
            base = f"https://huggingface.co/{repository}/resolve/{expected_revision}"
            config_hash = sha256(request_bytes(f"{base}/config.json", token))
            tokenizer_hash = sha256(request_bytes(f"{base}/tokenizer_config.json", token))
            config_hashes.add(config_hash)
            tokenizer_hashes.add(tokenizer_hash)
        rows.append(
            {
                "id": checkpoint["id"],
                "source_ref": source_ref,
                "expected_revision": expected_revision,
                "observed_revision": observed_revision,
                "revision_matches": observed_revision == expected_revision,
                "declared_stage": checkpoint["stage"],
                "stage_from_reported_token_boundary": stage_from_position,
                "stage_matches": checkpoint["stage"] == stage_from_position,
                "config_sha256": config_hash,
                "tokenizer_config_sha256": tokenizer_hash,
            }
        )

    audit = manifest["release_audit"]
    expected_config_prefix = audit["selected_config_sha256_prefix"]
    expected_tokenizer_prefix = audit["selected_tokenizer_config_sha256_prefix"]
    uniform_config = len(config_hashes) == 1 and next(iter(config_hashes)).startswith(
        expected_config_prefix
    )
    uniform_tokenizer = len(tokenizer_hashes) == 1 and next(iter(tokenizer_hashes)).startswith(
        expected_tokenizer_prefix
    )
    ready = (
        all(row["revision_matches"] and row["stage_matches"] for row in rows)
        and uniform_config
        and uniform_tokenizer
    )
    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "repository": repository,
        "observed_branch_count": len(refs.get("branches", [])),
        "observed_tag_count": len(refs.get("tags", [])),
        "selected_checkpoint_count": len(rows),
        "all_selected_refs_match": all(row["revision_matches"] for row in rows),
        "all_stages_match_reported_8b_boundaries": all(row["stage_matches"] for row in rows),
        "uniform_config": uniform_config,
        "uniform_tokenizer_config": uniform_tokenizer,
        "runnable_pending_observer_parity_smoke": ready,
        "status": "PASS" if ready else "FAIL",
        "checkpoints": rows,
        "note": (
            "PASS verifies release identity and static configuration only. It does not replace "
            "decoder-stack inspection, observer parity, behavioral validation, or artifact validation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = evaluate(manifest, os.environ.get("HF_TOKEN"))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_ready and report["status"] != "PASS":
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
