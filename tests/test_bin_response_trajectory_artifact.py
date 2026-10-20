from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bin_response_trajectory_artifact",
    ROOT / "scripts" / "bin_response_trajectory_artifact.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestRelativeResponseBinning(unittest.TestCase):
    def _artifact(self, path: Path, *, batch_size: int = 1, capped: bool = False) -> None:
        records = []
        for sample_index, sample_id in enumerate(("knowledge:test:0",) if batch_size == 1 else ("a", "b")):
            records.append(
                {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "task_name": "knowledge",
                    "split_name": "test",
                    "layer_index": 0,
                    "stream": "residual_state",
                    "module_name": "layers.0",
                    "phase": "prefill",
                    "decode_aggregation": "all_updates",
                    "decode_step": None,
                    "num_updates": 1,
                    "residual_delta_exact": True,
                    "vector": torch.tensor([0.0, 1.0]),
                }
            )
            for step in range(8):
                updated = dict(records[-1])
                updated.update(
                    {
                        "phase": "decode",
                        "decode_step": step,
                        "vector": torch.tensor([float(step), float(step + 1)]),
                    }
                )
                records.append(updated)
        torch.save(
            {
                "metadata": {
                    "decode_aggregation": "all_updates",
                    "decode_max_updates": 8 if capped else None,
                    "num_records": len(records),
                },
                "decode_aggregation": "all_updates",
                "decode_max_updates": 8 if capped else None,
                "truncated": {},
                "records": records,
                "num_records": len(records),
            },
            path,
        )

    def test_bins_every_uncapped_update_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.pt"
            output = root / "binned.pt"
            report = root / "report.json"
            self._artifact(source)
            result = MODULE.bin_artifact(source, output, report, bins=4)
            artifact = torch.load(output, map_location="cpu", weights_only=False)
            decode = [row for row in artifact["records"] if row["phase"] == "decode"]
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["response_update_count"]["per_sample"], {"knowledge:test:0": 8})
            self.assertEqual([row["num_updates"] for row in decode], [2, 2, 2, 2])
            self.assertEqual(
                [row["trajectory_position"] for row in decode],
                [f"relative_bin_{index:04d}" for index in range(4)],
            )
            self.assertTrue(torch.equal(decode[0]["vector"], torch.tensor([0.5, 1.5])))

    def test_rejects_batched_or_capped_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, kwargs, message in (
                ("batched", {"batch_size": 2}, "batch size one"),
                ("capped", {"capped": True}, "uncapped"),
            ):
                source = root / f"{name}.pt"
                self._artifact(source, **kwargs)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.bin_artifact(source, root / f"{name}-out.pt", root / f"{name}.json", bins=4)


if __name__ == "__main__":
    unittest.main()
