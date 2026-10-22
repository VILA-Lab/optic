#!/usr/bin/env python3
"""Generate a fixed token-only stream from the checkpoint and its tokenizer."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / "configs/calibration/self_generated.json"
sys.path.insert(0, str(ROOT / "scripts"))
from check_disk_space import evaluate_storage_policy  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maximum_identical_run(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [values.size]))
    return int(np.diff(boundaries).max())


def check_storage(design: dict[str, Any], phase: str, reserve_bytes: int = 0) -> dict[str, Any]:
    policy = json.loads(Path(design["storage"]["policy"]).read_text(encoding="utf-8"))
    report = evaluate_storage_policy(
        policy, reserve_bytes=reserve_bytes, reserve_scope="corpus", phase=phase
    )
    if not report["ok"]:
        raise RuntimeError(f"storage policy failed at {phase}: {report['checks']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    design = json.loads(args.design.read_text(encoding="utf-8"))
    if design.get("status") != "fixed":
        raise SystemExit("checkpoint self-generation design is not fixed")
    generation = design["generation"]
    if sha256_file(Path(__file__).resolve()) != generation["builder_sha256"]:
        raise SystemExit("self-generation builder hash differs from the design")
    model_config = design["model"]
    vocab = int(model_config["vocab_size"])
    sequence_count = int(generation["accepted_sequence_count"])
    sequence_length = int(generation["sequence_length"])
    new_tokens = int(generation["new_tokens_per_sequence"])
    batch_size = int(generation["batch_size"])
    if sequence_length != new_tokens + 2:
        raise SystemExit("sequence length must equal BOS + seed token + generated tokens")
    output_dir = args.output_dir or Path(design["storage"]["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    token_path = output_dir / "checkpoint_self_generated_token_ids_le_u32.bin"
    manifest_path = output_dir / "checkpoint_self_generated_token_manifest.json"
    if token_path.exists() or manifest_path.exists():
        raise SystemExit("refusing to overwrite checkpoint-self-generated token assets")
    expected_bytes = sequence_count * sequence_length * np.dtype("<u4").itemsize
    check_storage(design, "pre_checkpoint_self_generation", expected_bytes + 1024**3)

    import torch
    import transformers
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    snapshot = Path(model_config["local_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=True
    )
    special_ids = [int(value) for value in tokenizer.all_special_ids]
    if special_ids != design["tokenizer"]["all_special_ids"]:
        raise RuntimeError("tokenizer special IDs differ from the design")
    bos_id = int(tokenizer.bos_token_id)
    if bos_id != int(design["tokenizer"]["bos_token_id"]):
        raise RuntimeError("tokenizer BOS ID differs from the design")
    special_set = set(special_ids)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    torch.manual_seed(int(generation["sampling_seed"]))
    torch.cuda.manual_seed_all(int(generation["sampling_seed"]))
    token_stream = np.empty((sequence_count, sequence_length), dtype="<u4")
    accepted = 0
    candidate_index = 0
    rejected_unique = 0
    rejected_run = 0
    rejected_special = 0
    started = time.time()
    bad_words_ids = [[value] for value in special_ids]
    try:
        with torch.inference_mode():
            while accepted < sequence_count:
                candidate_seed_ids: list[int] = []
                candidate_indices: list[int] = []
                while len(candidate_seed_ids) < batch_size and candidate_index < int(generation["maximum_candidate_sequences"]):
                    seed_id = (
                        candidate_index * int(generation["seed_token_lcg_multiplier"])
                        + int(generation["seed_token_lcg_increment"])
                    ) % vocab
                    index = candidate_index
                    candidate_index += 1
                    if seed_id in special_set:
                        continue
                    candidate_seed_ids.append(seed_id)
                    candidate_indices.append(index)
                if not candidate_seed_ids:
                    break
                inputs = torch.empty(
                    (len(candidate_seed_ids), 2), dtype=torch.long, device=args.device
                )
                inputs[:, 0] = bos_id
                inputs[:, 1] = torch.tensor(
                    candidate_seed_ids, dtype=torch.long, device=args.device
                )
                attention_mask = torch.ones_like(inputs)
                generated = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=float(generation["temperature"]),
                    top_p=float(generation["top_p"]),
                    top_k=int(generation["top_k"]),
                    min_new_tokens=new_tokens,
                    max_new_tokens=new_tokens,
                    bad_words_ids=bad_words_ids,
                    pad_token_id=int(tokenizer.eos_token_id),
                ).to(device="cpu", dtype=torch.int64).numpy()
                if generated.shape != (len(candidate_seed_ids), sequence_length):
                    raise RuntimeError(f"generated token shape differs: {generated.shape}")
                for row in generated:
                    if accepted >= sequence_count:
                        break
                    generated_part = row[2:]
                    if any(int(value) in special_set for value in generated_part):
                        rejected_special += 1
                        continue
                    if np.unique(row).size < int(generation["minimum_unique_ids_per_sequence"]):
                        rejected_unique += 1
                        continue
                    if maximum_identical_run(row) > int(generation["maximum_identical_token_run"]):
                        rejected_run += 1
                        continue
                    token_stream[accepted] = row.astype("<u4", copy=False)
                    accepted += 1
                del generated, attention_mask, inputs
                print(
                    json.dumps(
                        {
                            "accepted_sequences": accepted,
                            "target_sequences": sequence_count,
                            "candidate_sequences": candidate_index,
                            "elapsed_seconds": time.time() - started,
                        }
                    ),
                    flush=True,
                )
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if accepted != sequence_count:
        raise RuntimeError(f"accepted only {accepted}/{sequence_count} self-generated sequences")
    if int(token_stream.max()) >= vocab:
        raise RuntimeError("self-generated stream contains an out-of-vocabulary ID")

    with token_path.open("wb") as handle:
        handle.write(memoryview(token_stream).cast("B"))
    if token_path.stat().st_size != expected_bytes:
        raise RuntimeError("self-generated token file byte count differs")
    flattened = token_stream.reshape(-1)
    non_special = flattened[~np.isin(flattened, np.asarray(special_ids, dtype=np.uint32))]
    unique_non_special = int(np.unique(non_special).size)
    minimum_coverage = int(generation["minimum_unique_non_special_token_ids"])
    if unique_non_special < minimum_coverage:
        raise RuntimeError(
            f"self-generated token coverage gate failed: {unique_non_special} < {minimum_coverage}"
        )
    manifest = {
        "schema_version": "ple-checkpoint-self-generated-token-stream-v1",
        "protocol_id": design["protocol_id"],
        "complete": True,
        "artifact_role": "checkpoint_and_tokenizer_only_fitting_tokens",
        "quality_evaluation_allowed": False,
        "external_text_or_benchmark_examples_used": False,
        "raw_text_persisted": False,
        "design": str(args.design.resolve()),
        "design_sha256": sha256_file(args.design),
        "model_identifier": model_config["identifier"],
        "model_revision": model_config["revision"],
        "checkpoint_sha256": model_config["weights_sha256"],
        "tokenizer_snapshot": str(snapshot.resolve()),
        "tokenizer_special_ids": special_ids,
        "token_file": str(token_path.resolve()),
        "token_file_sha256": sha256_file(token_path),
        "token_file_bytes": token_path.stat().st_size,
        "token_dtype": "uint32_little_endian",
        "sequence_count": sequence_count,
        "sequence_length": sequence_length,
        "positions": int(token_stream.size),
        "unique_non_special_token_ids": unique_non_special,
        "unique_non_special_token_fraction": unique_non_special / vocab,
        "complete_tokenizer_row_coverage_is_separate": True,
        "generation": generation,
        "candidate_sequences_seen": candidate_index,
        "rejections": {
            "special_token_in_generated_part": rejected_special,
            "insufficient_unique_ids": rejected_unique,
            "identical_token_run": rejected_run,
        },
        "device": args.device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    check_storage(design, "post_checkpoint_self_generation")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
