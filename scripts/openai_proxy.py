#!/usr/bin/env python3
"""
Round-robin OpenAI proxy for LightRAG multi-key load balancing.

Distributes LLM calls across multiple OpenAI API keys, effectively
multiplying your TPM budget by the number of keys.

Usage:
  1. Set OPENAI_KEY_1, OPENAI_KEY_2, OPENAI_KEY_3 in LightRAG/.env
  2. Run: python scripts/openai_proxy.py
  3. In LightRAG/.env set: LLM_BINDING_HOST=http://localhost:8080
  4. Restart LightRAG server

Requirements: pip install fastapi uvicorn httpx python-dotenv
"""

import itertools
import os
import sys
from pathlib import Path

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Load keys from LightRAG/.env
env_path = Path(__file__).parent.parent / "LightRAG" / ".env"
load_dotenv(env_path)

OPENAI_API_BASE = "https://api.openai.com"

keys = [
    os.environ.get("OPENAI_KEY_1", "").strip(),
    os.environ.get("OPENAI_KEY_2", "").strip(),
    os.environ.get("OPENAI_KEY_3", "").strip(),
]
active_keys = [k for k in keys if k]

if not active_keys:
    print("ERROR: No keys found. Set OPENAI_KEY_1 / OPENAI_KEY_2 / OPENAI_KEY_3 in LightRAG/.env")
    sys.exit(1)

print(f"Proxy started with {len(active_keys)} key(s) — round-robin across all requests")
print(f"Listening on http://localhost:8080")

_key_cycle = itertools.cycle(active_keys)
call_counts = {k[-6:]: 0 for k in active_keys}  # track by last 6 chars of key

app = FastAPI()


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    key = next(_key_cycle)
    call_counts[key[-6:]] += 1

    body = await request.body()
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": request.headers.get("Content-Type", "application/json"),
    }
    if "Accept" in request.headers:
        headers["Accept"] = request.headers["Accept"]

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.request(
            method=request.method,
            url=f"{OPENAI_API_BASE}/v1/{path}",
            content=body,
            headers=headers,
        )

    if resp.status_code == 429:
        # Log rate limit hits so you can tune MAX_ASYNC
        print(f"  [429] Rate limit on key ...{key[-6:]} — consider lowering MAX_ASYNC")

    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.get("/health")
async def health():
    return {"status": "ok", "keys": len(active_keys), "calls": call_counts}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
