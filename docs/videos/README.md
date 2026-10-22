# OPTIC video explainer

| Language | Video | Subtitles |
| --- | --- | --- |
| English | [2:56 walkthrough](optic-core-method-en.mp4) | [English SRT](optic-en.srt) |
| Chinese | [3:28 walkthrough](optic-core-method-zh.mp4) | [Chinese SRT](optic-zh.srt) |

Both videos are 1080p at 30 fps, with narration, captions and seven chapters.
They open with a 1.2-second paper-title cover ([image](cover.png),
[editable source](cover.svg)). Narration, captions and chapters shift together.
They follow the paper's motivation: embedding capacity scales alongside
Transformer computation. PLE accounts for 45.8% of Gemma 4 E2B parameters and
35.2% of E4B parameters.

The walkthrough explains how OPTIC pools a token's code budget across layers,
reconstructs each layer's PLE vector, and fits the projected PLE contribution
under cached gates and frozen output projections. The fit point is before
normalization and residual addition.

The closing results are for BF16 E4B: about 100× PLE compression retains 97.8%
of aggregate performance across seven benchmarks. Inference time is essentially
unchanged on an RTX 4090 at the same KV-cache setting. Compression figures
include codes, codebooks and decoder parameters and cover PLE storage.

File sizes, durations and SHA-256 checksums are in [manifest.json](manifest.json).

Narration is synthesized locally with
[Qwen3-TTS 1.7B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice):
**Aiden** for English and **Uncle_Fu** for Chinese.
Speech starts with each caption. Subtitles and chapters follow the timing of
each version. All 48 narration segments were checked against the encoded audio.
