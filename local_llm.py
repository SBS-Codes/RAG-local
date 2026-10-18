"""
Local LLM Module
=================

Provides a modular, reusable interface for running a local GGUF language model
using llama-cpp-python. This keeps the LLM logic completely separate from the
RAG pipeline so you can swap models, change parameters, or replace the backend
without touching the rest of the code.

Supports two modes:
  1. Direct GGUF loading via llama-cpp-python (no server needed)
  2. LM Studio / Ollama compatible OpenAI-style API (if server is running)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


# ── Default model path ────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    "/Users/shashibetadur/.lmstudio/models/"
    "lmstudio-community/Llama-3.2-3B-Instruct-GGUF/"
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)


class LocalLLM:
    """
    Wrapper for a local GGUF language model loaded via llama-cpp-python.

    Usage:
        llm = LocalLLM()                         # loads the default model
        answer = llm.generate("What is Mars?")    # returns plain text
        answer = llm.chat(system_prompt, user_prompt)  # chat-style call
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        """
        Loads the GGUF model into memory.

        :param model_path:    Absolute path to a .gguf file.
        :param n_ctx:         Context window size in tokens (default 2048).
        :param n_gpu_layers:  Layers to offload to GPU (0 = CPU only).
        :param verbose:       Whether llama.cpp prints internal logs.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download a GGUF model or update the path."
            )

        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "Install it with:  pip install llama-cpp-python"
            )

        print(f"Loading local LLM from: {path.name} ...")
        self.model = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        self.model_path = str(path)
        self.model_name = path.stem
        print(f"✅ Local LLM ready: {self.model_name}  (context={n_ctx} tokens)")

    # ── Simple text completion ────────────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.1,
        stop: Optional[list] = None,
    ) -> str:
        """
        Raw text completion. Sends the full prompt as-is to the model.
        """
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            echo=False,
        )
        return output["choices"][0]["text"].strip()

    # ── Chat-style completion (system + user) ─────────────────────────────────
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Chat-style completion using llama-cpp-python's chat API.
        Returns a dict with 'answer' and 'usage' keys.
        """
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        answer = response["choices"][0]["message"]["content"].strip()
        usage = response.get("usage", {})
        return {
            "answer": answer,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }


class LMStudioClient:
    """
    Alternative client that talks to LM Studio's local OpenAI-compatible server.
    Use this if you prefer to run the model in LM Studio's UI and call its API.

    Requires LM Studio to be running with "Start Server" enabled (default: port 1234).

    Usage:
        client = LMStudioClient()
        result = client.chat(system_prompt, user_prompt)
    """

    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.1,
        model: str = "local-model",
    ) -> Dict[str, Any]:
        """Sends a chat completion request to LM Studio's local API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required: pip install openai")

        client = OpenAI(base_url=self.base_url, api_key="lm-studio")
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response.choices[0].message.content.strip()
        usage = response.usage
        return {
            "answer": answer,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None,
        }


# ── Convenience factory ──────────────────────────────────────────────────────
def get_llm(
    mode: str = "direct",
    model_path: str = DEFAULT_MODEL_PATH,
    **kwargs,
):
    """
    Factory function to create an LLM client.

    :param mode:
        "direct"   → load GGUF file via llama-cpp-python (default, no server)
        "lmstudio" → connect to LM Studio's local OpenAI-compatible API
    :param model_path: Path to .gguf file (only used in "direct" mode).
    :returns: LocalLLM or LMStudioClient instance.
    """
    if mode == "direct":
        return LocalLLM(model_path=model_path, **kwargs)
    elif mode == "lmstudio":
        base_url = kwargs.get("base_url", "http://localhost:1234/v1")
        return LMStudioClient(base_url=base_url)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'direct' or 'lmstudio'.")
