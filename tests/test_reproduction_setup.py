import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionSetupTests(unittest.TestCase):
    def test_ifbench_scorer_receives_repository_absolute_path(self):
        from ple_eval import task_runner
        from ple_eval.config import load_benchmark, load_model

        benchmark = load_benchmark("ifbench")
        calls = []
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(task_runner, "run_logged", side_effect=lambda command, **kwargs: calls.append((command, kwargs))), \
                patch.object(task_runner.subprocess, "check_output", return_value=benchmark["postprocess"]["revision"] + "\n") as revision:
            task_runner.run_evalscope(benchmark, 8300, Path(directory), {}, False, load_model())
        scorer, options = next((command, kwargs) for command, kwargs in calls
                               if "rescore_ifbench_official_v2.py" in " ".join(command))
        repository = Path(scorer[scorer.index("--official-repo") + 1])
        self.assertEqual(repository, ROOT / benchmark["postprocess"]["repository"])
        self.assertTrue(repository.is_absolute())
        self.assertEqual(Path(revision.call_args.args[0][2]), repository)
        self.assertEqual(options["cwd"], ROOT / "evaluation_client")

    def test_local_paths_preserve_scientific_values(self):
        setup = load_script("configure_paths")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            config = root / "configs" / "model.json"
            config.write_text(json.dumps({"snapshot": "<MODEL_DIR>/gemma", "seed": 0, "shape": [4, 8]}))
            changed = setup.configure(root, {"model_dir": root / "models"})
            self.assertEqual(changed, ["configs/model.json"])
            self.assertEqual(json.loads(config.read_text()), {"snapshot": str(root / "models/gemma"), "seed": 0, "shape": [4, 8]})

    def test_disk_reserve_and_explicit_limits(self):
        storage = load_script("check_disk_space")
        policy = {"filesystem_path": ".", "corpus_cache_paths": [], "new_output_paths": []}
        filesystem = SimpleNamespace(f_bavail=100, f_frsize=1)
        with patch.object(storage.os, "statvfs", return_value=filesystem):
            self.assertTrue(storage.evaluate_storage_policy(policy, reserve_bytes=60)["ok"])
            self.assertFalse(storage.evaluate_storage_policy(policy, reserve_bytes=101)["ok"])
            policy["maximum_new_output_bytes"] = 50
            self.assertFalse(storage.evaluate_storage_policy(policy, reserve_bytes=60)["ok"])
            policy["maximum_new_output_bytes"] = None
            policy["maximum_corpus_cache_bytes"] = 50
            self.assertFalse(storage.evaluate_storage_policy(policy, reserve_bytes=60, reserve_scope="corpus")["ok"])


if __name__ == "__main__":
    unittest.main()
