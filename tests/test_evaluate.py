import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from ple_eval import cli, config, evaluate as main

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EvaluateTests(unittest.TestCase):
    def test_cli_defaults_select_main_method(self):
        with patch("sys.argv", ["ple-eval", "evaluate"]), \
                patch.object(cli, "evaluate_artifact", return_value={"ok": True}) as run, \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exit:
            cli.main()
        self.assertEqual(exit.exception.code, 0)
        run.assert_called_once_with(model_dir=None, artifact=None, gpus=[0], run_id=None,
                                    smoke=False, prepare_only=False)

    def test_run_selects_only_main_benchmarks_and_smoke_has_no_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(main, "ROOT", root), \
                    patch.object(main, "prepare", return_value={"model_config": "model.json", "environment": {}}), \
                    patch.object(main, "run_level", return_value={"ok": True}) as run, \
                    patch.object(main, "aggregate") as aggregate:
                result = main.evaluate_artifact(run_id="smoke", smoke=True)
            self.assertTrue(result["ok"])
            self.assertNotIn("result", result)
            aggregate.assert_not_called()
            self.assertEqual(run.call_count, 2)
            for call in run.call_args_list:
                self.assertEqual(call.kwargs["method_names"], [main.METHOD])
                self.assertEqual(call.kwargs["requested_benchmarks"], main.BENCHMARKS[call.kwargs["level_name"]])
                self.assertTrue(call.kwargs["smoke"])
            self.assertTrue((root / "results/smoke/smoke.json").is_file())
            self.assertFalse((root / "results/smoke/result.json").exists())

    def test_prepare_only_does_not_run_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "ROOT", Path(directory)), \
                    patch.object(main, "prepare", return_value={}), patch.object(main, "run_level") as run:
                result = main.evaluate_artifact(run_id="prepared", prepare_only=True)
            self.assertFalse(result["executed"])
            run.assert_not_called()

    def test_artifact_download_uses_pinned_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            downloaded = Path(directory) / "compressed_ple.npz"
            downloaded.write_bytes(b"released artifact")
            manifest = {"artifact": {"repo_id": "owner/model", "revision": "fixed-revision",
                                     "filename": "compressed_ple.npz", "sha256": config.file_sha256(downloaded)}}
            with patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)) as download:
                self.assertEqual(main.resolve_artifact(None, manifest), downloaded)
            download.assert_called_once_with(repo_id="owner/model", filename="compressed_ple.npz",
                                             revision="fixed-revision")

    def test_local_artifact_accepts_release_and_rejects_other_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release, other = (root / name for name in ("release.npz", "other.npz"))
            release.write_bytes(b"released checkpoint package")
            other.write_bytes(b"other fitted weights")
            manifest = {"artifact": {"sha256": config.file_sha256(release)}}
            self.assertEqual(main.resolve_artifact(release, manifest), release)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                main.resolve_artifact(other, manifest)

    def test_preparation_binds_exact_inputs_without_rewriting_tracked_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "input/compressed_ple.npz"
            artifact.parent.mkdir()
            artifact.write_bytes(b"released compressed PLE")
            snapshot = root / "model"
            snapshot.mkdir()
            (snapshot / "model.safetensors").write_bytes(b"weights")
            runtime = root / "bin"
            runtime.mkdir()
            (runtime / "vllm").touch()
            vllm_package = root / "site-packages/vllm"
            gemma4_source = vllm_package / "model_executor/models/gemma4.py"
            gemma4_source.parent.mkdir(parents=True)
            gemma4_source.write_text("original Gemma implementation\n")
            runtime_versions = {name: f"installed-{name}" for name in ("torch", "vllm", "transformers", "evalscope")}
            model = {"model_id": "google/gemma-4-E4B-it", "revision": "fixed-revision", "snapshot": "unconfigured",
                     "weight_file": "model.safetensors", "weight_sha256": config.file_sha256(snapshot / "model.safetensors"),
                     "runtime": {"python": "python", "vllm": "vllm"}}
            method = {"method_id": main.METHOD, "compressed_artifact_sha256": config.file_sha256(artifact),
                      "reconstruction_sha256": "unused", "audit_chunk_rows": 8192,
                      "training_metrics_path": "not-needed-for-evaluation", "training_metrics_sha256": "old"}
            for name in ("builder", "roundtrip_verifier", "online_adapter"):
                path = root / f"{name}.py"
                path.write_text("pass\n")
                method[f"{name}_path"] = path.name
                method[f"{name}_sha256"] = "old"
            recon = root / "artifacts/optic/e4b-bf16-r100-seed0/dense_ple_reconstruction.safetensors"
            recon.parent.mkdir(parents=True)
            recon.write_bytes(b"reconstructed")
            method["reconstruction_sha256"] = config.file_sha256(recon)
            model_config = root / "configs/models/e4b_bf16.json"
            method_config = root / f"configs/methods/{main.METHOD}.json"
            manifest = root / "configs/artifact.json"
            write(model_config, model)
            write(method_config, method)
            write(manifest, {"method": main.METHOD, "artifact": {"path": str(artifact),
                            "sha256": config.file_sha256(artifact)}})
            before = {path: path.read_bytes() for path in (model_config, method_config)}
            child_methods = []

            def command(command, **kwargs):
                child_methods.append(config.load_method(main.METHOD))

            with patch.object(main, "ROOT", root), patch.object(main, "MANIFEST", manifest), \
                    patch.object(main.sys, "executable", str(runtime / "python")), \
                    patch.object(main.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(vllm_package / "__init__.py"))), \
                    patch.object(main.importlib.metadata, "version", side_effect=runtime_versions.__getitem__), \
                    patch.object(main.subprocess, "run", side_effect=command):
                prepared = main.prepare(run_dir=root / "results/run", model_dir=snapshot, artifact=artifact, gpu=2)
            bound_model = config.read_json(Path(prepared["model_config"]))
            bound_method = config.read_json(Path(prepared["method_config"]))
            self.assertEqual(bound_model["snapshot"], str(snapshot))
            self.assertEqual(bound_model["runtime"]["python"], str(runtime / "python"))
            self.assertEqual(bound_method["compressed_artifact_path"], str(artifact))
            self.assertEqual(bound_method["compressed_artifact_sha256"], config.file_sha256(artifact))
            self.assertEqual(bound_method["builder_sha256"], config.file_sha256(root / "builder.py"))
            self.assertNotIn("training_metrics_path", bound_method)
            self.assertEqual(child_methods, [bound_method])
            self.assertEqual(prepared["runtime"]["package_versions"], runtime_versions)
            self.assertEqual(prepared["runtime"]["gemma4_source_sha256"], config.file_sha256(gemma4_source))
            for path, original in before.items():
                self.assertEqual(path.read_bytes(), original)

    def _completed_run(self, root):
        run = root / "results/actual"
        helpers = main._paper_helpers()
        rows = json.loads((ROOT / "docs/figures/main-table.json").read_text())
        row = next(row for row in rows if row["row"] == "OPTIC" and row["column"] == "E4B BF16")
        write(root / "docs/figures/main-table.json", [row])
        scores = {name: value / 100 for name, value in row["fits"][0]["scores_pct"].items()}
        for level, benchmarks in main.BENCHMARKS.items():
            write(run / level.lower() / "plan.json", {"smoke": False})
            for benchmark in benchmarks:
                task = run / level.lower() / main.METHOD / benchmark
                record = config.read_json(ROOT / "configs/benchmarks" / f"{benchmark}.json")
                write(task / "benchmark.json", record)
                write(task / "status.json", {"status": "completed"})
                write(task / "method.json", {"ple_compression_ratio": row["mean_actual_compression_ratio"]})
                write(task / "selected-model-config.json", {"model_id": "google/gemma-4-E4B-it", "revision": "fixed"})
                if benchmark == "l1_quality_bundle":
                    write(task / "full/summary.json", {"wikitext": {"perplexity": 12.0},
                          "piqa": {"accuracy": scores["piqa_accuracy"]}, "gsm8k": {"accuracy": scores["gsm8k_accuracy"]}})
                elif benchmark in {"gpqa_diamond", "mmlu_redux_30x100"}:
                    metric = "gpqa_accuracy" if benchmark == "gpqa_diamond" else "mmlu_redux_accuracy"
                    write(task / "full/summary.json", {"result": {"accuracy": scores[metric]}})
                else:
                    integrity = {"status": "passed"}
                    if benchmark == "ifbench":
                        write(task / "author-score.json", {"metrics": {"prompt_level_loose": scores["ifbench_prompt_loose"]}})
                    elif benchmark == "infographic_vqa":
                        report = task / "report.json"
                        write(report, {"score": scores["infographic_vqa_anls"]})
                        integrity["report_files"] = [str(report)]
                    else:
                        review = task / "review.jsonl"
                        review.write_text("\n".join(json.dumps({"sample_score": {"score": {"value": {"acc": int(i < 58)}}}})
                                                     for i in range(120)))
                        integrity.update(review_files=[str(review)], review_rows=120)
                    write(task / "evalscope-integrity.json", integrity)
        return run, helpers, row

    def test_aggregate_reads_actual_results_and_caps_each_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, helpers, row = self._completed_run(root)
            with patch.object(main, "ROOT", root), patch.object(main, "_paper_helpers", return_value=helpers):
                result = main.aggregate(run)
            self.assertAlmostEqual(result["retention_pct"], row["mean_retention_pct"])
            self.assertGreater(result["scores_pct"]["aime_accuracy"], result["reference"]["scores_pct"]["aime_accuracy"])
            summary = run / "l1" / main.METHOD / "l1_quality_bundle/full/summary.json"
            changed = config.read_json(summary)
            changed["piqa"]["accuracy"] = 0.1
            write(summary, changed)
            with patch.object(main, "ROOT", root), patch.object(main, "_paper_helpers", return_value=helpers):
                changed_result = main.aggregate(run)
            self.assertEqual(changed_result["scores_pct"]["piqa_accuracy"], 10.0)
            self.assertLess(changed_result["retention_pct"], result["retention_pct"])

    def test_aggregate_rejects_smoke_and_incomplete_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, helpers, _ = self._completed_run(root)
            with patch.object(main, "ROOT", root), patch.object(main, "_paper_helpers", return_value=helpers):
                write(run / "l1/plan.json", {"smoke": True})
                with self.assertRaisesRegex(ValueError, "smoke"):
                    main.aggregate(run)
                write(run / "l1/plan.json", {"smoke": False})
                write(run / "l2" / main.METHOD / "gpqa_diamond/status.json", {"status": "failed"})
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    main.aggregate(run)


if __name__ == "__main__":
    unittest.main()
