# Third-party material

## vLLM

`serving_overlay/vllm/model_executor/models/gemma4.py` is adapted from vLLM
0.19.1. Copyright contributors to the vLLM project; copyright 2025 The vLLM
team and Google Inc. / HuggingFace Inc. team. The local changes support PLE
reconstruction and the Gemma 4 checkpoint variants used in this comparison.

The [upstream source](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/gemma4.py)
is licensed under [Apache 2.0](LICENSES/Apache-2.0.txt). The accompanying patch
contains the changes to that source.

## GSM8K

`data/request-sets/gsm8k-fewshot-train-first-8.jsonl` and
`data/request-sets/gsm8k-full-reference-1319.jsonl` contain examples from
[OpenAI's GSM8K dataset](https://github.com/openai/grade-school-math).
Copyright (c) 2021 OpenAI. The [MIT license](LICENSES/GSM8K-MIT.txt) applies
to those examples.

## PIQA

`data/request-sets/piqa-validation-complete-1838.jsonl` contains the PIQA
validation set by Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao,
and Yejin Choi. The [authors' dataset page](https://ybisk.com/piqa/)
specifies the [Academic Free License 3.0](LICENSES/AFL-3.0.txt).

## WikiText-2

`data/request-sets/wikitext-2-raw-v1-test.txt` and
`data/calibration/e4b-calibration/calibration_text.txt` contain WikiText-2
text from Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher's
[WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext).
The calibration text uses the training split at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.

The [dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
attributes the text to Wikipedia and describes [CC BY-SA 4.0](LICENSES/CC-BY-SA-4.0.txt)
in its licensing section; its metadata also lists
[CC BY-SA 3.0](LICENSES/CC-BY-SA-3.0.txt) and [GFDL](LICENSES/GFDL-1.3.txt). The texts retain the
WikiText-2 content; request and calibration files select the splits and
positions specified by the configurations.

## Gemma checkpoints

The configurations refer to Google's Gemma 4 checkpoints, which are released
under the [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license).
Download checkpoints from the model repositories named in
[`configs/models/`](configs/models).
