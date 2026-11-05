"""LLM clients. Default: a local model through Ollama.

An OpenAI-compatible API (OpenAI, Groq, a vLLM server on Oracle Cloud, ...)
can be used instead by setting LLM_BACKEND=openai, LLM_BASE_URL, LLM_API_KEY
and LLM_MODEL. `ScriptedLLM` replays fixed answers for the tests.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import requests

SEED = 42


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class OllamaLLM:
    def __init__(self, model: str | None = None, host: str | None = None, num_ctx: int = 8192):
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.num_ctx = num_ctx

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def chat(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            # num_predict caps the answer length: a normal answer is ~200 tokens, and the cap
            # stops a rare runaway generation from blocking the app for minutes.
            "options": {"temperature": 0, "seed": SEED, "num_ctx": self.num_ctx, "num_predict": 1024},
        }
        if schema:
            body["format"] = schema
        t0 = time.perf_counter()
        for attempt in range(2):  # one retry if Ollama does not answer in time
            try:
                r = requests.post(f"{self.host}/api/chat", json=body, timeout=180)
                break
            except requests.exceptions.Timeout:
                if attempt == 1:
                    raise
        r.raise_for_status()
        data = r.json()
        return LLMResult(
            text=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            seconds=time.perf_counter() - t0,
        )


class OpenAICompatibleLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")

    @property
    def name(self) -> str:
        return f"api:{self.model}"

    def chat(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "seed": SEED,
        }
        if schema:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=300,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            seconds=time.perf_counter() - t0,
        )


class ScriptedLLM:
    """Test double: returns queued responses (dicts are sent as JSON)."""

    name = "scripted"

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        self.calls.append((system, user))
        item = self.responses.pop(0) if self.responses else {"abstain": True, "answer": "", "claims": []}
        text = item if isinstance(item, str) else json.dumps(item)
        return LLMResult(text=text, prompt_tokens=len(user) // 4, output_tokens=len(text) // 4, seconds=0.0)


class ExtractiveMockLLM:
    """No-model stand-in for UI checks: quotes the first sentence of passage C1.

    Select it with LLM_BACKEND=mock. It is never used for evaluation results.
    """

    name = "mock (extractive, no LLM)"

    def chat(self, system: str, user: str, schema: dict | None = None) -> LLMResult:
        import re

        m = re.search(r"\[C1\][^\n]*\n(.+?)(?:\n\n\[C2\]|\n\nQuestion:)", user, re.S)
        if not m:
            return LLMResult(text=json.dumps({"abstain": True, "answer": "", "claims": [], "explanation": "", "conflicts": ""}))
        body = re.sub(r"^\(?\d{1,3}[.)]\s*", "", m.group(1).strip())
        sentence = re.split(r"(?<=[a-z][.;])\s", body)[0][:300]
        data = {
            "abstain": False,
            "answer": f"{sentence} [C1]",
            "claims": [{"text": sentence, "citations": ["C1"], "quote": sentence[:150]}],
            "explanation": "Mock mode: the first sentence of the best passage is quoted.",
            "conflicts": "",
        }
        return LLMResult(text=json.dumps(data))


def make_llm():
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "openai":
        return OpenAICompatibleLLM()
    if backend == "mock":
        return ExtractiveMockLLM()
    return OllamaLLM()
