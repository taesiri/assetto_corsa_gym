from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from qwen_loader import count_parameters, load_qwen, model_dimensions


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=96, ge=1, le=512)


def build_app(
    *,
    model_name: str,
    fallback_model_name: str | None,
    device: str,
    load_mode: str,
) -> FastAPI:
    loaded = load_qwen(
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        device=device,
        load_mode=load_mode,
    )
    params = count_parameters(loaded.model)
    metadata = {
        "requested_model_name": model_name,
        "loaded_model_name": loaded.model_name,
        "fallback_used": loaded.fallback_used,
        "device": loaded.device,
        "load_mode": loaded.load_mode,
        "quantized_4bit": loaded.quantized_4bit,
        "model_type": getattr(loaded.config, "model_type", None),
        "hidden_size": model_dimensions(loaded.config)[0],
        "num_hidden_layers": model_dimensions(loaded.config)[1],
    }
    metadata.update(params)

    app = FastAPI(title="Qwen 3.5 Runtime", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **metadata}

    @app.get("/metadata")
    def get_metadata() -> dict[str, Any]:
        return metadata

    @app.post("/generate")
    def generate(request: GenerateRequest) -> dict[str, Any]:
        encoded = loaded.tokenizer(request.prompt, return_tensors="pt").to(loaded.model.device)
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            pad_token_id=loaded.tokenizer.eos_token_id,
        )
        text = loaded.tokenizer.decode(generated[0], skip_special_tokens=True)
        if text.startswith(request.prompt):
            text = text[len(request.prompt):].strip()
        return {"text": text, "model_name": loaded.model_name}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a dedicated Qwen 3.5 runtime from a Python 3.10 uv environment.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B-Base")
    parser.add_argument("--fallback-model-name", default="Qwen/Qwen3.5-2B-Base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-mode", choices=["auto", "4bit", "fp16", "cpu"], default="auto")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = build_app(
        model_name=args.model_name,
        fallback_model_name=args.fallback_model_name,
        device=args.device,
        load_mode=args.load_mode,
    )
    uvicorn.run(app, host=args.bind, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
