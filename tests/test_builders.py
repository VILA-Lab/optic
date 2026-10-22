import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

class ReleaseRegressionChecks(unittest.TestCase):
    def test_pca_pq_accepts_shipped_template_and_preserves_existing_outputs(self):
        builder = module("build_weighted_pca_pq_e2b")
        config = json.loads((ROOT / "configs/fit/weighted_pca_pq_e2b_bf16.json").read_text())
        with tempfile.TemporaryDirectory() as work:
            base = Path(work)
            weights = base / "model.safetensors"
            weights.write_bytes(b"model header fixture")
            model = config["source_model"]
            model["snapshot"] = str(base)
            model["weight_file_bytes"] = weights.stat().st_size
            for key, value in config["calibration"].items():
                if isinstance(value, str) and value.startswith("data/"):
                    config["calibration"][key] = str(ROOT / value)
            for key in ["source_builder", "wrapper_builder", "method_config_output"]:
                config[key] = str(ROOT / config[key])
            config["persistent_output_root"] = str(base / "persistent")
            config["capability_output_root"] = str(base / "capability")
            path = base / "config.json"
            path.write_text(json.dumps(config))
            real_digest = builder.sha256_file
            def fixture_digest(candidate):
                return model["weight_file_sha256"] if candidate == weights else real_digest(candidate)
            tensor = (model["ple_dtype"], model["ple_shape"], model["ple_dense_bytes"])
            with patch.object(builder, "sha256_file", side_effect=fixture_digest), patch.object(builder, "tensor_contract", return_value=tensor):
                self.assertTrue(Path(config["method_config_output"]).is_file())
                builder.validate(path)
                Path(config["persistent_output_root"]).mkdir()
                with self.assertRaisesRegex(SystemExit, "persistent output root already exists"):
                    builder.validate(path)
                Path(config["persistent_output_root"]).rmdir()
                Path(config["capability_output_root"]).mkdir()
                with self.assertRaisesRegex(SystemExit, "capability output root already exists"):
                    builder.validate(path)

    def test_beam_encoder_accepts_template_and_preserves_partial_outputs(self):
        builder = module("build_aqlm_w4a16")
        class BoundaryReached(Exception):
            pass
        with tempfile.TemporaryDirectory() as work:
            base = Path(work)
            paths = {name: base / name for name in ["root", "logs", "stages", "method_config", "block_reconstruction", "compact", "build_receipt"]}
            paths["method_config"].write_text("{}")
            config = {"models": {"e2b": {}}}
            with patch.object(builder, "validate_config"), patch.object(builder, "output_paths", return_value=paths), patch.object(builder, "verify_matched_parent_child_ple", side_effect=BoundaryReached):
                with self.assertRaises(BoundaryReached):
                    builder.run_cell(config, "e2b", "cuda:0")
                paths["compact"].write_bytes(b"partial fit fixture")
                with self.assertRaisesRegex(RuntimeError, "outputs already exist"):
                    builder.run_cell(config, "e2b", "cuda:0")

if __name__ == "__main__":
    unittest.main(verbosity=2)
