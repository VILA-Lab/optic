"""Serve an evaluation runtime through the EvalScope client."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .config import canonical_json, resolve_model_config, write_json_atomic
from .direct_common import load_llm, shutdown_llm
from .evaluation_runtime import validate_runtime_binding, validate_runtime_interface


def create_app(*, model_config_path: Path, max_model_len: int, seed: int, output_dir: Path,
               startup_id: str | None = None) -> FastAPI:
    config_path = resolve_model_config(model_config_path)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError('The model configuration must be an object.')
    binding = config.get("evaluation_runtime")
    validate_runtime_binding(binding)
    validate_runtime_interface(binding, "chat")
    selected_config = {'path': str(config_path), 'sha256': hashlib.sha256(config_bytes).hexdigest()}
    startup_id = str(uuid.UUID(startup_id)) if startup_id is not None else str(uuid.uuid4())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        output_dir.mkdir(parents=True, exist_ok=False)
        runtime = None
        audit = None
        try:
            runtime, load_seconds = await run_in_threadpool(
                load_llm, config["snapshot"], max_model_len=max_model_len, seed=seed,
                dtype=config.get("dtype", "bfloat16"), load_format=config.get("load_format", "safetensors"),
                quantization=config.get("quantization"), evaluation_runtime=binding)
            audit = (output_dir / "requests.jsonl").open("x", encoding="utf-8")
            write_json_atomic(output_dir / "runtime.json", {
                "model_config": selected_config, "engine_load_seconds": load_seconds,
                "runtime_metadata": getattr(runtime, 'execution_metadata', None),
                "startup_id": startup_id,
            })
            app.state.runtime = runtime
            app.state.request_lock = asyncio.Lock()
            app.state.audit = audit
            yield
        finally:
            if audit is not None:
                audit.close()
            if runtime is not None:
                shutdown_llm(runtime)
            app.state.runtime = None

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        if getattr(app.state, "runtime", None) is None:
            raise HTTPException(status_code=503, detail="Evaluation runtime is unavailable.")
        return {"status": "ready", "startup_id": startup_id,
                "model_config_sha256": selected_config['sha256']}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": config["served_model_name"], "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def complete(request: Request):
        arrived = time.monotonic()
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError) as error:
            raise HTTPException(status_code=400, detail="Invalid JSON request.") from error
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="A chat request must be an object.")
        request_digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        async with app.state.request_lock:
            started = time.monotonic()
            row = {"request_json_sha256": request_digest,
                   "queued_elapsed_seconds": started - arrived}
            try:
                response, runtime_record = await run_in_threadpool(app.state.runtime.complete_chat, payload)
                row.update({"request_status": "completed", "runtime": runtime_record, "response": response})
                return response
            except ValueError as error:
                row.update({"request_status": "rejected", "error": str(error)})
                raise HTTPException(status_code=400, detail=str(error)) from error
            except Exception as error:
                row.update({"request_status": "failed", "error": f"{type(error).__name__}: {error}"})
                raise HTTPException(status_code=500, detail="Evaluation model execution failed.") from error
            finally:
                row["execution_elapsed_seconds"] = time.monotonic() - started
                app.state.audit.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                app.state.audit.flush()

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--startup-id", required=True)
    args = parser.parse_args()
    if args.max_model_len <= 0 or not 0 < args.port < 65536:
        parser.error("The context limit and local server port must be positive and valid.")
    import uvicorn
    uvicorn.run(create_app(model_config_path=args.model_config, max_model_len=args.max_model_len,
                          seed=args.seed, output_dir=args.output_dir, startup_id=args.startup_id),
                host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
