#!/usr/bin/env python3
"""EvalScope prompt override for InfographicVQA."""

from __future__ import annotations

import json
import os
from typing import Any


def _apply_infovqa_lmms_eval() -> None:
    """Match the pinned lmms-eval InfographicVQA prompt exactly."""
    from evalscope.api.dataset import Sample
    from evalscope.api.messages import ChatMessageUser, ContentImage, ContentText
    from evalscope.benchmarks.infovqa import infovqa_adapter as module
    from evalscope.utils.io_utils import bytes_to_base64

    prompt = "{question}\nAnswer the question using a single word or phrase."

    def record_to_sample(self: Any, record: dict[str, Any]) -> Sample:
        content = []
        image = record.get("image")
        if image:
            content.append(ContentImage(image=bytes_to_base64(image["bytes"], format="png", add_header=True)))
        content.append(ContentText(text=prompt.format(question=record["question"])))
        return Sample(
            input=[ChatMessageUser(content=content)],
            target=json.dumps(record.get("answers")),
            metadata={
                "questionId": record.get("questionId"),
                "answer_type": record.get("answer_type"),
                "image_url": record.get("image_url"),
                "ocr": record.get("ocr"),
                "protocol_prompt": "lmms_eval_infovqa_val_fba26a24",
                "image_first": True,
            },
        )

    module.PROMPT = prompt
    module.InfoVQAAdapter.record_to_sample = record_to_sample


def apply_from_environment() -> None:
    override = os.environ.get("GEMMA4_BENCHMARK_OVERRIDE", "none")
    if override == "none":
        return
    if override == "infovqa_lmms_eval":
        _apply_infovqa_lmms_eval()
        return
    raise ValueError(f"Unknown GEMMA4_BENCHMARK_OVERRIDE={override!r}")
