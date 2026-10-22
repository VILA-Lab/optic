"""Shared invariants for active PLE baseline builders."""

from __future__ import annotations

PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
RECON_KEY = "embed_tokens_per_layer.weight"
VOCAB_SIZE = 262144
MAX_UINT16_CODE_COUNT = 65536


def require_uint16_code_count(name: str, count: int) -> None:
    if not (2 <= int(count) <= MAX_UINT16_CODE_COUNT):
        raise SystemExit(f"{name} must be in [2, {MAX_UINT16_CODE_COUNT}] because resident codes are uint16-packed")


def load_ple_tensor(safe_open, path: str, num_layers: int, d_ple: int, *, allow_recon_key: bool = False):
    expected_shape = (VOCAB_SIZE, num_layers * d_ple)
    allowed_keys = [PLE_KEY]
    if allow_recon_key:
        allowed_keys.append(RECON_KEY)
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        key = next((candidate for candidate in allowed_keys if candidate in keys), None)
        if key is None:
            raise SystemExit(f"missing PLE tensor key in {path}; expected one of {allowed_keys}")
        tensor = f.get_tensor(key)
    if tuple(tensor.shape) != expected_shape:
        raise SystemExit(f"PLE tensor shape {tuple(tensor.shape)} != {expected_shape} for key {key}")
    return tensor, key, expected_shape
