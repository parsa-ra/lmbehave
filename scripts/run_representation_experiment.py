#!/usr/bin/env python3
"""Plan, run, validate, and analyze a manifest-driven representation experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "representation_dynamics_v1.json"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
SAMPLE_SELECTION_FILENAME = "sample_selection.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id_set_sha256(sample_ids: Iterable[str]) -> str:
    """Hash sample identity independently of evaluation/batch traversal order."""

    return hashlib.sha256("\n".join(sorted(set(sample_ids))).encode()).hexdigest()


def git_state(path: Path) -> dict[str, Any]:
    def output(*args: str) -> bytes:
        return subprocess.run(("git", *args), cwd=path, check=True, stdout=subprocess.PIPE).stdout

    status = output("status", "--porcelain=v1", "--untracked-files=all")
    fingerprint = None
    if status:
        digest = hashlib.sha256(status)
        digest.update(output("diff", "--binary", "HEAD", "--"))
        for line in status.decode("utf-8", errors="surrogateescape").splitlines():
            if line.startswith("?? "):
                candidate = path / line[3:]
                if candidate.is_file():
                    digest.update(line[3:].encode("utf-8", errors="surrogateescape"))
                    digest.update(candidate.read_bytes())
        fingerprint = digest.hexdigest()
    return {
        "head": output("rev-parse", "HEAD").decode().strip(),
        "branch": output("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
        "dirty": bool(status),
        "source_fingerprint_sha256": fingerprint,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Only experiment manifest schema_version=1 is supported")
    if not manifest.get("models") or not manifest.get("panels"):
        raise ValueError("Manifest must define models and panels")
    for panel_name, panel in manifest["panels"].items():
        knowledge = [item["task"] for item in panel.get("knowledge_tasks", [])]
        reasoning = [item["task"] for item in panel.get("reasoning_tasks", [])]
        if len(knowledge) < 2 or len(reasoning) < 2:
            raise ValueError(f"Panel {panel_name!r} needs at least two tasks per class")
        if len(knowledge) != len(reasoning):
            raise ValueError(f"Panel {panel_name!r} must be class-balanced at task level")
        if len(set(knowledge + reasoning)) != len(knowledge + reasoning):
            raise ValueError(f"Panel {panel_name!r} contains duplicate task names")
        reference = panel.get("reference_positive_tasks")
        if reference is not None:
            if len(reference) * 2 != len(knowledge) + len(reasoning):
                raise ValueError(f"Panel {panel_name!r} reference partition must contain half the tasks")
            if not set(reference) <= set(knowledge + reasoning):
                raise ValueError(f"Panel {panel_name!r} reference partition contains unknown tasks")
        assignments = math.comb(len(knowledge) + len(reasoning), len(knowledge))
        for setting in (
            "linear_task_label_permutations",
            "fdr_task_label_permutations",
            "nonlinear_task_label_permutations",
        ):
            requested = int(manifest["analysis"].get(setting, 0))
            if requested < 0 or requested > assignments - 1:
                raise ValueError(
                    f"Panel {panel_name!r} has only {assignments - 1} non-observed balanced assignments, "
                    f"but {setting} requests {requested}"
                )
    return manifest


def selected(mapping: dict[str, Any], requested: list[str] | None, label: str) -> list[str]:
    names = list(mapping)
    if not requested:
        return names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return [name for name in names if name in requested]


def task_items(panel: dict[str, Any]):
    for class_name, key in (("knowledge", "knowledge_tasks"), ("reasoning", "reasoning_tasks")):
        for item in panel[key]:
            yield class_name, item


def scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def model_args(model: dict[str, Any]) -> str:
    values = {"pretrained": model["pretrained"], "revision": model["revision"], **model.get("model_args", {})}
    return ",".join(f"{key}={scalar(value)}" for key, value in values.items())


def task_root(run_dir: Path, panel_name: str, model_name: str, class_name: str, task_name: str) -> Path:
    return run_dir / panel_name / model_name / "tasks" / f"{class_name}-{task_name}"


def artifact_for(root: Path) -> Path:
    artifacts = sorted((root / "representations").glob("*/representations.pt"))
    if len(artifacts) != 1:
        raise RuntimeError(f"Expected one representation artifact under {root}, found {len(artifacts)}")
    return artifacts[0]


def result_for(root: Path) -> Path:
    results = sorted((root / "evaluation").glob("*/*_results.json"))
    if len(results) != 1:
        raise RuntimeError(f"Expected one results JSON under {root}, found {len(results)}")
    return results[0]


def completed_task_output(root: Path) -> tuple[Path, Path] | None:
    """Return a complete task output, reject partial or ambiguous output."""
    artifacts = sorted((root / "representations").glob("*/representations.pt"))
    results = sorted((root / "evaluation").glob("*/*_results.json"))
    if not artifacts and not results:
        return None
    if len(artifacts) != 1 or len(results) != 1:
        raise RuntimeError(
            f"Incomplete or ambiguous task output under {root}: "
            f"representations={len(artifacts)}, results={len(results)}"
        )
    return artifacts[0], results[0]


def build_command(
    python: Path,
    run_dir: Path,
    panel_name: str,
    panel: dict[str, Any],
    model_name: str,
    model: dict[str, Any],
    class_name: str,
    task_name: str,
    extraction: dict[str, Any],
    limit: int,
) -> list[str]:
    root = task_root(run_dir, panel_name, model_name, class_name, task_name)
    command = [
        str(python),
        "-m",
        "lmms_eval",
        "--model",
        model["lmms_model"],
        "--model_args",
        model_args(model),
    ]
    if not panel.get("use_task_generation_kwargs", False):
        command.extend(("--gen_kwargs", f"max_new_tokens={panel['max_new_tokens']},temperature=0.0"))
    command.extend(
        [
        "--tasks",
        task_name,
        "--log_samples",
        "--output_path",
        str(root / "evaluation"),
        "--batch_size",
        str(model["batch_size"]),
        "--limit",
        str(limit),
        "--seed",
        str(extraction["seed"]),
        "--representation_dynamics",
        "--representation_dynamics_output",
        str(root / "representations"),
        "--representation_dynamics_pooling",
        extraction["pooling"],
        "--representation_dynamics_phases",
        extraction["phases"],
        "--representation_dynamics_storage_dtype",
        extraction["storage_dtype"],
        ]
    )
    if "decode_aggregation" in extraction:
        command.extend(
            (
                "--representation_dynamics_decode_aggregation",
                str(extraction["decode_aggregation"]),
            )
        )
    if extraction.get("decode_max_updates") is not None:
        command.extend(
            (
                "--representation_dynamics_decode_max_updates",
                str(int(extraction["decode_max_updates"])),
            )
        )
    return command


def environment_metadata(python: Path) -> dict[str, Any]:
    code = """
import importlib.metadata
import json
import sys
import torch
import transformers
import lmms_eval

packages = {}
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name")
    if name:
        packages[name.lower().replace("_", "-")] = distribution.version
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "transformers": transformers.__version__,
    "transformers_import_path": transformers.__file__,
    "lmms_eval_import_path": lmms_eval.__file__,
    "packages": dict(sorted(packages.items())),
}))
"""
    metadata = json.loads(
        subprocess.run((str(python), "-c", code), check=True, stdout=subprocess.PIPE, text=True).stdout
    )
    package_bytes = json.dumps(metadata["packages"], sort_keys=True, separators=(",", ":")).encode()
    metadata["package_inventory_sha256"] = hashlib.sha256(package_bytes).hexdigest()
    return metadata


def resolved_run_manifest(manifest_path: Path, manifest: dict[str, Any], python: Path, limit: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_manifest": str(manifest_path.resolve()),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "experiment_id": manifest["experiment_id"],
        "effective_sample_count_per_task": limit,
        "root_git": git_state(ROOT),
        "lmms_eval_git": git_state(ROOT / "lmms-eval"),
        "environment": {"platform": platform.platform(), **environment_metadata(python)},
        "resolved_experiment": manifest,
    }


def selected_task_names(manifest: dict[str, Any], panel_names: list[str]) -> list[str]:
    """Return selected task names once, preserving manifest order."""
    names: list[str] = []
    seen: set[str] = set()
    for panel_name in panel_names:
        for _class_name, item in task_items(manifest["panels"][panel_name]):
            name = item["task"]
            if name not in seen:
                names.append(name)
                seen.add(name)
    return names


def build_sample_selection(
    manifest_path: Path,
    manifest: dict[str, Any],
    panel_names: list[str],
    limit: int,
) -> dict[str, Any]:
    """Resolve the exact first-N lmms-eval document identities before model loading."""
    from datasets.fingerprint import Hasher
    from lmms_eval.tasks import TaskManager

    manager = TaskManager(verbosity="ERROR")
    tasks: dict[str, Any] = {}
    for task_name in selected_task_names(manifest, panel_names):
        loaded = manager.load_task_or_group(task_name)
        if set(loaded) != {task_name}:
            raise RuntimeError(f"Expected one concrete task for {task_name!r}, loaded {sorted(loaded)}")
        task = loaded[task_name]
        if task.has_test_docs():
            split = task.config.test_split
            docs = task.test_docs()
        elif task.has_validation_docs():
            split = task.config.validation_split
            docs = task.validation_docs()
        else:
            raise RuntimeError(f"Task {task_name!r} has neither a test nor validation split")
        available = len(docs)
        if available < limit:
            raise RuntimeError(f"Task {task_name!r} has {available} documents, fewer than requested {limit}")
        selected_docs = docs.select(range(limit)) if hasattr(docs, "select") else list(docs)[:limit]
        sample_ids = [f"{task_name}:{split}:{index}" for index in range(limit)]
        tasks[task_name] = {
            "split": split,
            "available_documents": available,
            "selected_count": limit,
            "sample_ids": sample_ids,
            "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
            "evaluation_docs_fingerprint": getattr(docs, "_fingerprint", None),
            "selected_docs_fingerprint": Hasher.hash(selected_docs),
        }
    return {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "effective_sample_count_per_task": limit,
        "panels": panel_names,
        "root_git": git_state(ROOT),
        "lmms_eval_git": git_state(ROOT / "lmms-eval"),
        "tasks": tasks,
    }


def freeze_samples(
    run_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    panel_names: list[str],
    limit: int,
    allow_dirty: bool,
) -> None:
    candidate = build_sample_selection(manifest_path, manifest, panel_names, limit)
    if not allow_dirty and (candidate["root_git"]["dirty"] or candidate["lmms_eval_git"]["dirty"]):
        raise RuntimeError("Final sample selection requires clean root and lmms-eval trees")
    path = run_dir / SAMPLE_SELECTION_FILENAME
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in current.items() if key != "created_at_utc"}
        if comparable != candidate:
            raise RuntimeError(f"Existing {path} does not match the current task selection or source state")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        candidate["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "path": str(path),
                "tasks": len(candidate["tasks"]),
                "samples_per_task": limit,
                "sample_selection_sha256": sha256_file(path),
            },
            indent=2,
        )
    )


def bind_sample_selection(candidate: dict[str, Any], run_dir: Path, manifest: dict[str, Any]) -> None:
    if not manifest.get("require_frozen_sample_ids", False):
        return
    path = run_dir / SAMPLE_SELECTION_FILENAME
    if not path.is_file():
        raise RuntimeError(f"Missing {path}; run freeze-samples before extraction")
    selection = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment_id": candidate["experiment_id"],
        "experiment_manifest_sha256": candidate["experiment_manifest_sha256"],
        "effective_sample_count_per_task": candidate["effective_sample_count_per_task"],
        "root_git": candidate["root_git"],
        "lmms_eval_git": candidate["lmms_eval_git"],
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(f"Frozen sample selection has an unexpected {key}")
    expected_panels = list(manifest["panels"])
    if selection.get("panels") != expected_panels:
        raise RuntimeError("Frozen sample selection has an unexpected panel order or scope")
    expected_tasks = set(selected_task_names(manifest, expected_panels))
    if set(selection.get("tasks", {})) != expected_tasks:
        raise RuntimeError("Frozen sample selection does not cover every manifest task exactly once")
    for task_name, task_selection in selection["tasks"].items():
        sample_ids = task_selection.get("sample_ids")
        if not isinstance(sample_ids, list) or len(sample_ids) != candidate["effective_sample_count_per_task"]:
            raise RuntimeError(f"Frozen sample selection has an invalid count for {task_name}")
        if len(set(sample_ids)) != len(sample_ids):
            raise RuntimeError(f"Frozen sample selection contains duplicate IDs for {task_name}")
        digest = hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()
        if task_selection.get("sample_ids_sha256") != digest:
            raise RuntimeError(f"Frozen sample selection has an invalid ID digest for {task_name}")
        if not task_selection.get("selected_docs_fingerprint"):
            raise RuntimeError(f"Frozen sample selection lacks a document fingerprint for {task_name}")
    candidate["sample_selection"] = {"file": SAMPLE_SELECTION_FILENAME, "sha256": sha256_file(path)}


def load_bound_sample_selection(run_dir: Path, run_manifest: dict[str, Any]) -> dict[str, Any] | None:
    binding = run_manifest.get("sample_selection")
    if binding is None:
        return None
    if binding.get("file") != SAMPLE_SELECTION_FILENAME:
        raise RuntimeError("Run manifest contains an unexpected sample-selection path")
    path = run_dir / SAMPLE_SELECTION_FILENAME
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        raise RuntimeError("Frozen sample selection changed after extraction began")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_run_manifest(
    run_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    python: Path,
    limit: int,
    allow_dirty: bool,
) -> dict[str, Any]:
    candidate = resolved_run_manifest(manifest_path, manifest, python, limit)
    bind_sample_selection(candidate, run_dir, manifest)
    if not allow_dirty and (candidate["root_git"]["dirty"] or candidate["lmms_eval_git"]["dirty"]):
        raise RuntimeError("Final runs require clean root and lmms-eval trees; use --allow-dirty only for debugging")
    path = run_dir / "run_manifest.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        invariant_keys = (
            "experiment_manifest_sha256",
            "effective_sample_count_per_task",
            "root_git",
            "lmms_eval_git",
            "environment",
            "resolved_experiment",
        )
        if any(current.get(key) != candidate.get(key) for key in invariant_keys):
            raise RuntimeError(
                f"Existing {path} does not match the current manifest, sample count, source state, or environment"
            )
        return current
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return candidate


def plan(manifest: dict[str, Any], panel_names: list[str], model_names: list[str], limit: int) -> None:
    rows = []
    for panel_name in panel_names:
        panel = manifest["panels"][panel_name]
        task_count = len(panel["knowledge_tasks"]) + len(panel["reasoning_tasks"])
        assignments = math.comb(task_count, len(panel["knowledge_tasks"]))
        rows.append(
            {
                "panel": panel_name,
                "models": model_names,
                "tasks_per_class": len(panel["knowledge_tasks"]),
                "examples_per_task": limit,
                "model_task_processes": len(model_names) * task_count,
                "balanced_task_label_assignments": assignments,
                "complementary_equivalence_classes": assignments // 2,
                "minimum_class_symmetric_p": 2 / assignments,
                "claim_limit": panel["claim_limit"],
            }
        )
    print(json.dumps(rows, indent=2))


def run_tasks(
    manifest_path: Path,
    manifest: dict[str, Any],
    run_dir: Path,
    python: Path,
    panel_names: list[str],
    model_names: list[str],
    limit: int,
    allow_dirty: bool,
    dry_run: bool,
    requested_tasks: list[str] | None,
    skip_existing: bool,
) -> None:
    if not dry_run:
        ensure_run_manifest(run_dir, manifest_path, manifest, python, limit, allow_dirty)
    commands: list[tuple[Path, list[str]]] = []
    available_tasks = {
        item["task"] for panel_name in panel_names for _class_name, item in task_items(manifest["panels"][panel_name])
    }
    unknown_tasks = sorted(set(requested_tasks or ()) - available_tasks)
    if unknown_tasks:
        raise ValueError(f"Unknown task for selected panels: {', '.join(unknown_tasks)}")
    for panel_name in panel_names:
        panel = manifest["panels"][panel_name]
        for model_name in model_names:
            model = manifest["models"][model_name]
            for class_name, item in task_items(panel):
                if requested_tasks and item["task"] not in requested_tasks:
                    continue
                root = task_root(run_dir, panel_name, model_name, class_name, item["task"])
                existing = completed_task_output(root)
                if existing is not None:
                    if skip_existing:
                        continue
                    raise RuntimeError(f"Refusing to overwrite a completed task under {root}; use a new run directory")
                commands.append(
                    (root, build_command(
                        python,
                        run_dir,
                        panel_name,
                        panel,
                        model_name,
                        model,
                        class_name,
                        item["task"],
                        manifest["extraction"],
                        limit,
                    ))
                )
    if dry_run:
        for _root, command in commands:
            print(subprocess.list2cmdline(command))
        return
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for index, (root, command) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {subprocess.list2cmdline(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if completed_task_output(root) is None:
            raise RuntimeError(
                f"lmms-eval returned success without writing both evaluation and representation artifacts under {root}"
            )


def validation_command(python: Path, artifacts: list[Path], task_name: str, limit: int) -> list[str]:
    command = [
        str(python),
        str(ROOT / "scripts" / "validate_representation_artifact.py"),
        *(str(path) for path in artifacts),
        "--expected-task",
        f"{task_name}={limit}",
        "--require-exact",
    ]
    if len(artifacts) > 1:
        command.append("--require-matching-sample-ids")
    return command


def artifact_completion(root: Path, artifact: Path, task_name: str) -> dict[str, Any]:
    metadata = json.loads((artifact.parent / "metadata.json").read_text(encoding="utf-8"))["metadata"]
    sample_ids = set()
    with (artifact.parent / "records.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_ids.add(row["sample_id"])
    results_path = result_for(root)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    sample_id_digest = sample_id_set_sha256(sample_ids)
    return {
        "task": task_name,
        "artifact": str(artifact),
        "artifact_sha256": sha256_file(artifact),
        "result_json": str(results_path),
        "result_json_sha256": sha256_file(results_path),
        "task_hash": results.get("task_hashes", {}).get(task_name),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": sample_id_digest,
        "records": metadata.get("num_records"),
        "layers": metadata.get("num_layers"),
        "selected_stack": metadata.get("selected_stack"),
        "storage_dtype": metadata.get("storage_dtype"),
        "lmms_eval_git_head": metadata.get("lmms_eval_git_head"),
        "lmms_eval_git_hash": metadata.get("lmms_eval_git_hash"),
        "lmms_eval_git_dirty": metadata.get("lmms_eval_git_dirty"),
        "lmms_eval_source_fingerprint_sha256": metadata.get("lmms_eval_source_fingerprint_sha256"),
    }


def validate_run(
    manifest: dict[str, Any],
    run_dir: Path,
    python: Path,
    panel_names: list[str],
    model_names: list[str],
) -> None:
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise RuntimeError(f"Missing {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    sample_selection = load_bound_sample_selection(run_dir, run_manifest)
    locked_manifest = run_manifest.get("resolved_experiment")
    if not isinstance(locked_manifest, dict):
        raise RuntimeError("Run manifest does not contain a resolved experiment")
    for section in ("analysis", "extraction", "datasets"):
        if locked_manifest.get(section) != manifest.get(section):
            raise RuntimeError(f"Current experiment {section} differs from the run-locked manifest")
    for panel_name in panel_names:
        if locked_manifest.get("panels", {}).get(panel_name) != manifest["panels"][panel_name]:
            raise RuntimeError(f"Panel {panel_name} differs from the run-locked manifest")
    for model_name in model_names:
        if locked_manifest.get("models", {}).get(model_name) != manifest["models"][model_name]:
            raise RuntimeError(f"Model {model_name} differs from the run-locked manifest")
    limit = int(run_manifest["effective_sample_count_per_task"])
    validations = []
    completions = []
    for panel_name in panel_names:
        panel = manifest["panels"][panel_name]
        for class_name, item in task_items(panel):
            roots = [task_root(run_dir, panel_name, model_name, class_name, item["task"]) for model_name in model_names]
            artifacts = [artifact_for(root) for root in roots]
            result = subprocess.run(
                validation_command(python, artifacts, item["task"], limit),
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            validations.extend(json.loads(result.stdout))
            for model_name, root, artifact in zip(model_names, roots, artifacts):
                completion = artifact_completion(root, artifact, item["task"])
                completion.update({"panel": panel_name, "model_key": model_name, "class": class_name})
                expected_lmms = run_manifest["lmms_eval_git"]
                recorded_head = completion["lmms_eval_git_head"]
                legacy_hash = str(completion["lmms_eval_git_hash"] or "")
                head_matches = (
                    recorded_head == expected_lmms["head"]
                    if recorded_head is not None
                    else bool(legacy_hash) and expected_lmms["head"].startswith(legacy_hash)
                )
                if not head_matches:
                    raise RuntimeError(f"Artifact {artifact} was extracted from an unexpected lmms-eval commit")
                if bool(completion["lmms_eval_git_dirty"]) != bool(expected_lmms["dirty"]):
                    raise RuntimeError(f"Artifact {artifact} has an unexpected lmms-eval dirty state")
                if completion["lmms_eval_source_fingerprint_sha256"] != expected_lmms["source_fingerprint_sha256"]:
                    raise RuntimeError(f"Artifact {artifact} has an unexpected lmms-eval source fingerprint")
                if sample_selection is not None:
                    # lmms-eval may reorder requests by length for batching. Sample
                    # identity is therefore a set property, not an evaluation-order
                    # property. The stored freeze digest remains order-sensitive so
                    # the frozen selection file itself can detect reordering.
                    expected_ids = sample_id_set_sha256(
                        sample_selection["tasks"][item["task"]]["sample_ids"]
                    )
                    if completion["sample_ids_sha256"] != expected_ids:
                        raise RuntimeError(f"Artifact {artifact} does not match the frozen sample IDs")
                completions.append(completion)
    protocol_audit_path = run_dir / "protocol_audit.json"
    subprocess.run(
        (
            str(python),
            str(ROOT / "scripts" / "audit_text_protocol_samples.py"),
            str(run_dir),
            "--output",
            str(protocol_audit_path),
        ),
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    validation_source = {
        "root_git": git_state(ROOT),
        "lmms_eval_git": git_state(ROOT / "lmms-eval"),
    }
    report = {
        "status": "PASS",
        "validation_source": validation_source,
        "protocol_audit": {
            "file": str(protocol_audit_path),
            "sha256": sha256_file(protocol_audit_path),
        },
        "validations": validations,
    }
    (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed = {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "validation_source": validation_source,
        "artifacts": completions,
    }
    (run_dir / "completed_manifest.json").write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "artifacts": len(completions)}, indent=2))


def effective_analysis_settings(
    locked_settings: dict[str, Any], nonlinear_permutations: int | None
) -> dict[str, Any]:
    """Return analysis settings with one explicit, provenance-recorded sensitivity override."""

    settings = dict(locked_settings)
    if nonlinear_permutations is not None:
        if nonlinear_permutations < 0:
            raise ValueError("--nonlinear-permutations must be non-negative")
        settings["nonlinear_task_label_permutations"] = nonlinear_permutations
    return settings


def analyze_run(
    manifest: dict[str, Any],
    run_dir: Path,
    python: Path,
    panel_names: list[str],
    model_names: list[str],
    jobs: int,
    parallel_backend: str,
    analysis_backend: str,
    analysis_name: str,
    phase: str,
    nonlinear_permutations: int | None = None,
) -> None:
    validation_path = run_dir / "validation_report.json"
    completed_path = run_dir / "completed_manifest.json"
    run_manifest_path = run_dir / "run_manifest.json"
    if not validation_path.is_file() or not completed_path.is_file() or not run_manifest_path.is_file():
        raise RuntimeError("Run validation before analysis")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    load_bound_sample_selection(run_dir, run_manifest)
    if validation.get("status") != "PASS":
        raise RuntimeError("Validation report is not PASS")
    if completed.get("run_manifest_sha256") != sha256_file(run_manifest_path):
        raise RuntimeError("Completed manifest does not bind the current run manifest")
    locked_manifest = run_manifest.get("resolved_experiment")
    if not isinstance(locked_manifest, dict):
        raise RuntimeError("Run manifest does not contain a resolved experiment")
    for section in ("analysis", "extraction", "datasets"):
        if locked_manifest.get(section) != manifest.get(section):
            raise RuntimeError(f"Current experiment {section} differs from the run-locked manifest")
    for panel_name in panel_names:
        if locked_manifest.get("panels", {}).get(panel_name) != manifest["panels"][panel_name]:
            raise RuntimeError(f"Panel {panel_name} differs from the run-locked manifest")
    for model_name in model_names:
        if locked_manifest.get("models", {}).get(model_name) != manifest["models"][model_name]:
            raise RuntimeError(f"Model {model_name} differs from the run-locked manifest")
    completed_hashes = {entry["artifact"]: entry["artifact_sha256"] for entry in completed["artifacts"]}
    analysis = effective_analysis_settings(manifest["analysis"], nonlinear_permutations)
    for panel_name in panel_names:
        panel = manifest["panels"][panel_name]
        knowledge = [item["task"] for item in panel["knowledge_tasks"]]
        reasoning = [item["task"] for item in panel["reasoning_tasks"]]
        for model_name in model_names:
            artifacts = [
                artifact_for(task_root(run_dir, panel_name, model_name, class_name, item["task"]))
                for class_name, item in task_items(panel)
            ]
            for artifact in artifacts:
                if completed_hashes.get(str(artifact)) != sha256_file(artifact):
                    raise RuntimeError(f"Artifact is missing from validation or changed after validation: {artifact}")
            output_dir = run_dir / panel_name / model_name / analysis_name
            command = [
                str(python),
                "-m",
                "lmms_eval.representations.analysis",
                *(str(path) for path in artifacts),
                "--knowledge-tasks",
                ",".join(knowledge),
                "--reasoning-tasks",
                ",".join(reasoning),
                "--output-dir",
                str(output_dir),
                "--phase",
                phase,
                "--cv-folds",
                str(analysis["cv_folds"]),
                "--permutations",
                str(analysis["linear_task_label_permutations"]),
                "--fdr-permutations",
                str(
                    analysis.get(
                        "fdr_task_label_permutations",
                        analysis["linear_task_label_permutations"],
                    )
                ),
                "--nonlinear-permutations",
                str(analysis["nonlinear_task_label_permutations"]),
                "--pca-variance",
                str(analysis["pca_retained_variance"]),
                "--silhouette-max-samples",
                str(analysis["silhouette_max_samples"]),
                "--seed",
                str(analysis["seed"]),
                "--jobs",
                str(jobs),
                "--parallel-backend",
                parallel_backend,
                "--analysis-backend",
                analysis_backend,
            ]
            if panel.get("reference_positive_tasks"):
                command.extend(
                    (
                        "--reference-positive-tasks",
                        ",".join(panel["reference_positive_tasks"]),
                    )
                )
            print(subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            analysis_manifest = {
                "schema_version": 1,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_manifest_sha256": sha256_file(run_manifest_path),
                "completed_manifest_sha256": sha256_file(completed_path),
                "analysis_source": {
                    "root_git": git_state(ROOT),
                    "lmms_eval_git": git_state(ROOT / "lmms-eval"),
                },
                "settings": {
                    **analysis,
                    "jobs": jobs,
                    "parallel_backend": parallel_backend,
                    "analysis_backend": analysis_backend,
                    "phase": phase,
                },
                "settings_override": (
                    {"nonlinear_task_label_permutations": nonlinear_permutations}
                    if nonlinear_permutations is not None
                    else {}
                ),
                "artifacts": [{"path": str(path), "sha256": sha256_file(path)} for path in artifacts],
                "outputs": {
                    "layer_separability_csv_sha256": sha256_file(output_dir / "layer_separability.csv"),
                    "report_json_sha256": sha256_file(output_dir / "report.json"),
                    "permutation_nulls_json_gz_sha256": sha256_file(
                        output_dir / "permutation_nulls.json.gz"
                    ),
                },
            }
            (output_dir / "analysis_manifest.json").write_text(
                json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


def check_tasks(manifest: dict[str, Any], panel_names: list[str]) -> None:
    from lmms_eval.tasks import TaskManager

    available = set(TaskManager(verbosity="ERROR").all_tasks)
    requested = {item["task"] for panel_name in panel_names for _class, item in task_items(manifest["panels"][panel_name])}
    missing = sorted(requested - available)
    print(json.dumps({"requested": len(requested), "missing": missing}, indent=2))
    if missing:
        raise SystemExit(1)


def check_models(manifest: dict[str, Any], model_names: list[str]) -> None:
    """Verify that each pinned repository revision resolves to actual model weights."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    checks = []
    for model_name in model_names:
        model = manifest["models"][model_name]
        repository = model["pretrained"]
        revision = model["revision"]
        info = api.model_info(repository, revision=revision, files_metadata=False)
        files = [sibling.rfilename for sibling in (info.siblings or [])]
        weight_files = sorted(
            name
            for name in files
            if name.endswith((".safetensors", ".bin")) and not name.startswith(("adapter_", "optimizer"))
        )
        if info.sha != revision:
            raise RuntimeError(
                f"Model {model_name!r} resolved {repository}@{revision} to unexpected commit {info.sha}"
            )
        if not weight_files:
            raise RuntimeError(f"Model {model_name!r} has no model weight files at {repository}@{revision}")
        checks.append(
            {
                "model": model_name,
                "repository": repository,
                "revision": revision,
                "weight_files": len(weight_files),
            }
        )
    print(json.dumps({"status": "PASS", "models": checks}, indent=2))


def add_selectors(parser: argparse.ArgumentParser, *, run_dir: bool = False) -> None:
    parser.add_argument("--panel", action="append", help="Panel key; repeat to select multiple.")
    parser.add_argument("--model", action="append", help="Model key; repeat to select multiple.")
    if run_dir:
        parser.add_argument("--run-dir", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    add_selectors(plan_parser)
    plan_parser.add_argument("--limit", type=int)
    check_parser = commands.add_parser("check-tasks")
    add_selectors(check_parser)
    check_models_parser = commands.add_parser("check-models")
    add_selectors(check_models_parser)
    freeze_parser = commands.add_parser("freeze-samples")
    add_selectors(freeze_parser, run_dir=True)
    freeze_parser.add_argument("--limit", type=int)
    freeze_parser.add_argument("--allow-dirty", action="store_true")
    run_parser = commands.add_parser("run")
    add_selectors(run_parser, run_dir=True)
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--allow-dirty", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--task", action="append", help="Run only this task; repeat to select multiple.")
    run_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume by skipping tasks with exactly one representation artifact and one evaluation result.",
    )
    validate_parser = commands.add_parser("validate")
    add_selectors(validate_parser, run_dir=True)
    analyze_parser = commands.add_parser("analyze")
    add_selectors(analyze_parser, run_dir=True)
    analyze_parser.add_argument("--jobs", type=int, default=16, help="Concurrent layer/stream analysis workers.")
    analyze_parser.add_argument("--parallel-backend", choices=("process", "thread"), default="process")
    analyze_parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="sklearn")
    analyze_parser.add_argument(
        "--analysis-name",
        default="analysis",
        help="Basename for per-model analysis outputs; use a distinct name for each estimator family.",
    )
    analyze_parser.add_argument(
        "--phase",
        choices=("prefill", "decode"),
        default="prefill",
        help="Analyze prefill states or position-resolved/aggregated decode states.",
    )
    analyze_parser.add_argument(
        "--nonlinear-permutations",
        type=int,
        help=(
            "Override only the run-locked nonlinear task-label permutation count for a named "
            "sensitivity analysis. The effective value is recorded in analysis_manifest.json."
        ),
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    python = args.python.expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if not python.is_file():
        raise ValueError(f"Python executable does not exist: {python}")
    panel_names = selected(manifest["panels"], args.panel, "panel")
    model_names = selected(manifest["models"], args.model, "model")
    limit = int(getattr(args, "limit", None) or manifest["extraction"]["sample_count_per_task"])
    if limit <= 0:
        raise ValueError("Sample limit must be positive")

    if args.command == "plan":
        plan(manifest, panel_names, model_names, limit)
    elif args.command == "check-tasks":
        check_tasks(manifest, panel_names)
    elif args.command == "check-models":
        check_models(manifest, model_names)
    elif args.command == "freeze-samples":
        freeze_samples(
            args.run_dir.resolve(),
            manifest_path,
            manifest,
            panel_names,
            limit,
            args.allow_dirty,
        )
    elif args.command == "run":
        run_tasks(
            manifest_path,
            manifest,
            args.run_dir.resolve(),
            python,
            panel_names,
            model_names,
            limit,
            args.allow_dirty,
            args.dry_run,
            args.task,
            args.skip_existing,
        )
    elif args.command == "validate":
        validate_run(manifest, args.run_dir.resolve(), python, panel_names, model_names)
    elif args.command == "analyze":
        if args.jobs < 1:
            raise ValueError("--jobs must be at least 1")
        if Path(args.analysis_name).name != args.analysis_name or args.analysis_name in {"", ".", ".."}:
            raise ValueError("--analysis-name must be one non-empty path basename")
        analyze_run(
            manifest,
            args.run_dir.resolve(),
            python,
            panel_names,
            model_names,
            args.jobs,
            args.parallel_backend,
            args.analysis_backend,
            args.analysis_name,
            args.phase,
            args.nonlinear_permutations,
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
