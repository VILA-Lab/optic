# GPU inference

Run BF16 Gemma 4 E4B with the released OPTIC artifact. The example loads the
model directly onto one GPU and accepts either text or an image with a question.
Use Python 3.12 and a CUDA GPU; a 24 GiB GPU accommodates the model and short
generation examples below.

From the repository root:

```bash
python -m pip install -r requirements-inference.txt
python -m pip install -e .
python examples/generate.py --prompt "Explain why the sky is blue in two sentences."
```

The first run downloads the pinned model snapshot and the
[OPTIC artifact](https://huggingface.co/zcahjl3/gemma-4-E4B-it-OPTIC-100x);
subsequent runs reuse them. Existing files can be selected with
`--model-dir /path/to/snapshot --artifact /path/to/compressed_ple.npz`.

For a local image:

```bash
python examples/generate.py \
  --image /path/to/image.png \
  --prompt "Describe the image." \
  --device cuda:0 --max-new-tokens 128
```

The model's processor builds the image tokens and chat prompt. The vision encoder
feeds the language model through the normal Gemma interface, with OPTIC supplying
the language model's per-layer embeddings. The answer is printed to standard
output; load time, generation time, generated tokens, and GPU memory are printed
to standard error.

## vLLM

Install vLLM in the same environment and generate with it:

```bash
python -m pip install vllm==0.19.1
python examples/generate_vllm.py --prompt "Explain why the sky is blue in two sentences."
```

To serve an OpenAI-compatible endpoint, point `PLE_OPTIC_ARTIFACT` to the
artifact and start vLLM as usual:

```bash
export PLE_OPTIC_ARTIFACT=$(python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('zcahjl3/gemma-4-E4B-it-OPTIC-100x', 'compressed_ple.npz'))")
vllm serve google/gemma-4-E4B-it --revision ee0ef6023621cff504d758262d4e04895a5af4a2 --max-model-len 8192
```

The `ple_eval` vLLM plugin, installed by `pip install -e .`, builds the compact
decoder in place of the dense PLE table and skips the dense PLE tensor when it
loads the checkpoint. vLLM then reports about 10 GiB of model memory instead of
15.3 GiB with the dense PLE. The compact PLE runs with tensor parallel size 1.

## Loading and decoding

The example creates empty model parameters, installs OPTIC, and streams the
remaining checkpoint tensors directly to the GPU. It skips the original dense
PLE tensor before reading it from the checkpoint.

The default artifact keeps **56.4 MB** of codes, codebooks, and decoders resident,
replacing **5.64 GB** of BF16 PLE weights: **99.94×** PLE compression. During each
forward pass, the adapter looks up only the requested token codes and reconstructs
all layer embeddings with one stacked matrix multiplication. The multiplication
uses FP16 inputs with FP32 accumulation and bias addition, then returns BF16 rows.

The implementation is in
[`compact_adapter.py`](../src/ple_eval/compact_adapter.py). To attach it to an
already loaded Gemma model:

```python
from huggingface_hub import hf_hub_download
from ple_eval.compact_adapter import install_compact_ple

artifact = hf_hub_download("zcahjl3/gemma-4-E4B-it-OPTIC-100x", "compressed_ple.npz")
install_compact_ple(model, artifact, device="cuda:0")
```

[`examples/generate.py`](../examples/generate.py) also provides the streaming
`load_model` function for loading directly with compressed PLE. The
[evaluation guide](evaluation.md) runs the seven benchmarks through vLLM
with reconstructed PLE weights.
