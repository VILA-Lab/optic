"""Converter checks; run with llama.cpp/gguf-py and this repo's src on PYTHONPATH."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gguf
import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/convert_optic_gguf.py"
spec = importlib.util.spec_from_file_location("convert_optic_gguf", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class OpticGgufTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base = self.directory / "base.gguf"
        self.artifact = self.directory / "optic.npz"
        self.metadata = {
            "kind": "optic", "latent_mode": "discrete", "ple_shape": [5, 12],
            "d_ple": 4, "latent_dim": 8, "n_groups": 2, "codebook_size": 256,
            "group_dim": 4, "code_bits": 8, "num_layers": 3, "vocab_size": 5,
            "model_id": "google/gemma-4-E4B-it", "model_revision": "test-revision",
        }
        random = np.random.default_rng(7)
        self.arrays = {
            "token_codes_packed_uint8": np.array([0, 255, 128, 127, 1, 200, 254, 129, 255, 0], dtype=np.uint8),
            "codebooks_fp16": random.normal(size=(2, 256, 4)).astype(np.float16),
            "decoder_weight_fp16": random.normal(size=(3, 4, 8)).astype(np.float16),
            "decoder_bias_fp16": random.normal(size=(3, 4)).astype(np.float16),
        }
        self.write_artifact()
        self.write_base()

    def write_artifact(self):
        np.savez(self.artifact, metadata=np.array(json.dumps(self.metadata)), **self.arrays)

    def write_base(self, *, quantized_ple=False):
        writer = gguf.GGUFWriter(self.base, "gemma4")
        writer.add_uint32("gemma4.block_count", 3)
        writer.add_uint32("gemma4.embedding_length_per_layer_input", 4)
        writer.add_key_value("custom.empty", "", gguf.GGUFValueType.STRING)
        writer.add_key_value("custom.typed_array", [-1, 2], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.INT16)
        writer.add_key_value("custom.signed", -93, gguf.GGUFValueType.INT64)
        writer.add_key_value("custom.strings", ["one", "", "三"], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.STRING)
        writer.add_custom_alignment(64)
        weight = np.arange(96, dtype=np.float32).reshape(3, 32) / 20
        writer.add_tensor("blk.0.quantized", gguf.quants.quantize(weight, gguf.GGMLQuantizationType.Q4_0), raw_dtype=gguf.GGMLQuantizationType.Q4_0)
        bf16 = torch.arange(60).reshape(5, 12).to(torch.bfloat16).view(torch.uint16).numpy().view(np.uint8).reshape(5, 24)
        writer.add_tensor(converter.DENSE_NAME, bf16, raw_dtype=gguf.GGMLQuantizationType.BF16)
        writer.add_tensor("output_norm.weight", np.arange(4, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_compact_preserves_quantized_payload_metadata_and_unsigned_codes(self):
        output = self.directory / "compact.gguf"
        result = converter.convert(self.base, self.artifact, output)
        reader = gguf.GGUFReader(output)
        tensors = {t.name: t for t in reader.tensors}
        self.assertNotIn(converter.DENSE_NAME, tensors)
        self.assertEqual(tensors[converter.TENSOR_NAMES["codes"]].tensor_type, gguf.GGMLQuantizationType.I8)
        np.testing.assert_array_equal(tensors[converter.TENSOR_NAMES["codes"]].data.view(np.uint8).reshape(-1), self.arrays["token_codes_packed_uint8"])
        for name in ("codebooks", "decoder", "bias"):
            self.assertEqual(tensors[converter.TENSOR_NAMES[name]].tensor_type, gguf.GGMLQuantizationType.F16)
        self.assertEqual(result["compact_payload_bytes"], sum(a.nbytes for a in self.arrays.values()))
        self.assertEqual(reader.get_field("gemma4.optic.version").contents(), 1)
        self.assertTrue(result["verification"]["non_ple_payloads_identical"])
        self.assertEqual(reader.alignment, 64)
        self.assertEqual(reader.get_field("custom.typed_array").types, [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.INT16])

    def test_dense_control_matches_adapter_bf16_without_scale_for_boundary_tokens(self):
        from ple_eval.compact_adapter import load_compact_ple
        output = self.directory / "dense.gguf"
        converter.convert(self.base, self.artifact, output, mode="dense-reconstructed", chunk_tokens=2, threads=1)
        reader = gguf.GGUFReader(output)
        dense = next(t for t in reader.tensors if t.name == converter.DENSE_NAME)
        actual = torch.from_numpy(dense.data.copy()).view(torch.bfloat16).reshape(5, 12)
        self.metadata["compressed_ple_format_id"] = "optic_npz"
        self.write_artifact()
        module = load_compact_ple(self.artifact, device="cpu")
        ids = torch.tensor([0, 4, 2, 1])
        torch.testing.assert_close(actual[ids], module(ids), atol=0, rtol=0)
        self.assertIsNone(reader.get_field("gemma4.optic.version"))
        self.assertEqual(reader.get_field("gemma4.optic.control").contents(), "dense-reconstructed")

    def test_rejects_bad_geometry_dtype_and_hash(self):
        self.metadata["latent_dim"] = 9
        self.write_artifact()
        with self.assertRaisesRegex(ValueError, "inconsistent geometry"):
            converter.load_artifact(self.artifact)
        self.metadata["latent_dim"] = 8
        self.arrays["token_codes_packed_uint8"] = self.arrays["token_codes_packed_uint8"].astype(np.int16)
        self.write_artifact()
        with self.assertRaisesRegex(ValueError, "token_codes_packed_uint8"):
            converter.load_artifact(self.artifact)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            converter.convert(self.base, self.artifact, self.directory / "bad.gguf", expected_artifact_sha256="0" * 64)

    def test_rejects_base_shape_mismatch_and_existing_output(self):
        self.metadata["ple_shape"] = [6, 12]
        _, tensors = converter.load_artifact(self.artifact)
        with self.assertRaisesRegex(ValueError, "base PLE table shape"):
            converter.validate_base(gguf.GGUFReader(self.base), self.metadata)
        existing = self.directory / "exists.gguf"
        existing.touch()
        with self.assertRaises(FileExistsError):
            converter.convert(self.base, self.artifact, existing)
        with self.assertRaisesRegex(ValueError, "output must differ"):
            converter.convert(self.base, self.artifact, self.base)


    def test_cli_rejects_manifest_aliases_before_mutating_any_file(self):
        output = self.directory / "result.gguf"
        output.write_bytes(b"existing output must survive")
        protected = {"base": self.base, "artifact": self.artifact, "output": output}
        original = {name: path.read_bytes() for name, path in protected.items()}
        for target_name, target in protected.items():
            for alias_kind in ("relative", "symlink", "hardlink"):
                for force in (False, True):
                    with self.subTest(target=target_name, alias=alias_kind, force=force):
                        if alias_kind == "relative":
                            manifest = Path(os.path.relpath(target, Path.cwd()))
                        else:
                            manifest = self.directory / f"manifest-{target_name}-{alias_kind}-{force}.json"
                            if alias_kind == "symlink":
                                manifest.symlink_to(target)
                            else:
                                os.link(target, manifest)
                        argv = [str(SCRIPT), "--base", str(self.base), "--artifact", str(self.artifact),
                                "--output", str(output), "--manifest", str(manifest)]
                        if force:
                            argv.append("--force")
                        with mock.patch.object(sys, "argv", argv):
                            with self.assertRaisesRegex(ValueError, "manifest must differ"):
                                converter.main()
                        for name, path in protected.items():
                            self.assertEqual(path.read_bytes(), original[name])

    def test_default_manifest_alias_and_missing_output_alias_are_rejected(self):
        output = self.directory / "not-created" / "result.gguf"
        before = self.artifact.read_bytes()
        with self.assertRaisesRegex(ValueError, "manifest must differ from output"):
            converter.validate_destination_paths(self.base, self.artifact, output, manifest=output)
        self.assertFalse(output.parent.exists())
        output = self.directory / "result.gguf"
        default_manifest = Path(str(output) + ".json")
        default_manifest.symlink_to(self.artifact)
        argv = [str(SCRIPT), "--base", str(self.base), "--artifact", str(self.artifact), "--output", str(output)]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "manifest must differ from artifact"):
                converter.main()
        self.assertEqual(self.artifact.read_bytes(), before)
        self.assertFalse(output.exists())

    def test_output_aliases_both_inputs_are_rejected_even_with_force(self):
        for target_name, target in (("base", self.base), ("artifact", self.artifact)):
            before = target.read_bytes()
            for alias_kind in ("relative", "symlink", "hardlink"):
                with self.subTest(target=target_name, alias=alias_kind):
                    if alias_kind == "relative":
                        output = Path(os.path.relpath(target, Path.cwd()))
                    else:
                        output = self.directory / f"output-{target_name}-{alias_kind}.gguf"
                        if alias_kind == "symlink":
                            output.symlink_to(target)
                        else:
                            os.link(target, output)
                    with self.assertRaisesRegex(ValueError, "output must differ"):
                        converter.convert(self.base, self.artifact, output, force=True)
                    self.assertEqual(target.read_bytes(), before)

    def test_existing_manifest_requires_force_before_conversion(self):
        output = self.directory / "new.gguf"
        manifest = self.directory / "existing.json"
        manifest.write_bytes(b"existing manifest must survive")
        argv = [str(SCRIPT), "--base", str(self.base), "--artifact", str(self.artifact),
                "--output", str(output), "--manifest", str(manifest)]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(FileExistsError):
                converter.main()
        self.assertEqual(manifest.read_bytes(), b"existing manifest must survive")
        self.assertFalse(output.exists())
        converter.validate_destination_paths(self.base, self.artifact, output, manifest=manifest, force=True)


if __name__ == "__main__":
    unittest.main()
