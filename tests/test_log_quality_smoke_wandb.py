import ast
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "log_quality_smoke_wandb.py"


class WandbReporterSourceTests(unittest.TestCase):
    def test_fold_local_scope_maps_to_inductive_claim(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        strings = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertIn("inductive_fold_local_preprocessing", strings)
        self.assertNotIn("training_tasks_only_per_fold", strings)

    def test_task_warning_is_not_hardcoded_to_visual_panel(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('panel_spec["claim_limit"]', source)
        self.assertNotIn("Cross-benchmark labels confound modality", source)


if __name__ == "__main__":
    unittest.main()
