#!/usr/bin/env python3
"""Check whether Apertus 1.5 intermediate checkpoints are publicly discoverable."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "apertus15_longitudinal_v1.json"
CHECKPOINT_PATTERN = re.compile(r"(?:checkpoint|ckpt|stage|step|sft|rl)", re.IGNORECASE)


def request_json(url: str, token: str | None) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "vlmbehave-release-check/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def immutable_revision(value: Any) -> bool:
    text = str(value)
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def discover_candidates(
    refs: dict[str, Any], related_models: list[dict[str, Any]], endpoint: str
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for kind in ("branches", "tags"):
        for item in refs.get(kind, []):
            name = str(item.get("name", ""))
            revision = str(item.get("targetCommit", ""))
            if (
                name != "main"
                and CHECKPOINT_PATTERN.search(name)
                and immutable_revision(revision)
            ):
                candidates.append(
                    {
                        "kind": kind[:-2] if kind.endswith("es") else kind[:-1],
                        "name": name,
                        "revision": revision,
                    }
                )
    for item in related_models:
        model_id = str(item.get("id", ""))
        revision = str(item.get("sha", ""))
        if (
            model_id != endpoint
            and CHECKPOINT_PATTERN.search(model_id)
            and immutable_revision(revision)
        ):
            candidates.append(
                {"kind": "repository", "name": model_id, "revision": revision}
            )
    return sorted(
        candidates, key=lambda item: (item["kind"], item["name"], item["revision"])
    )


def evaluate_release(
    manifest: dict[str, Any], refs: dict[str, Any], related_models: list[dict[str, Any]]
) -> dict[str, Any]:
    endpoint = manifest["released_endpoint"]
    main_refs = [
        item for item in refs.get("branches", []) if item.get("name") == "main"
    ]
    observed_main = main_refs[0].get("targetCommit") if len(main_refs) == 1 else None
    candidates = discover_candidates(refs, related_models, endpoint["repository"])
    required = int(manifest["release_gate"]["minimum_intermediate_checkpoint_refs"])
    ready = observed_main == endpoint["revision"] and len(candidates) >= required
    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "endpoint_repository": endpoint["repository"],
        "expected_main_revision": endpoint["revision"],
        "observed_main_revision": observed_main,
        "main_revision_matches_manifest": observed_main == endpoint["revision"],
        "candidate_checkpoint_refs": candidates,
        "candidate_count": len(candidates),
        "minimum_candidate_refs_required": required,
        "longitudinal_ready": ready,
        "status": "READY_FOR_MANUAL_TRAJECTORY_AUDIT"
        if ready
        else "PENDING_INTERMEDIATE_CHECKPOINTS",
        "note": (
            "Candidate names are only a discovery aid. Before promotion, verify that checkpoints share "
            "one documented training trajectory, architecture, model code, and a common position unit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    endpoint = manifest["released_endpoint"]["repository"]
    token = os.environ.get("HF_TOKEN")
    refs = request_json(f"https://huggingface.co/api/models/{endpoint}/refs", token)
    related = request_json(
        "https://huggingface.co/api/models?author=swiss-ai&search=Apertus-v1.5&limit=100",
        token,
    )
    report = evaluate_release(manifest, refs, related)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_ready and not report["longitudinal_ready"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
