import os
from typing import Any


def call_llm(prompt: str, *, provider: str | None = None, **kwargs: Any) -> str:
    """Return a response from an LLM backend.

    The backend is chosen via the ``provider`` argument or the ``LLM_PROVIDER``
    environment variable. ``provider`` must be ``"openai"`` or ``"llama"``.

    For OpenAI, the :mod:`openai` package must be installed and an API key must
    be available in the ``OPENAI_API_KEY`` environment variable. Additional
    keyword arguments such as ``model`` are forwarded to
    ``client.chat.completions.create``.

    For Llama, the :mod:`llama_cpp` package must be installed. A ``model_path``
    pointing to the local model weights is required. Keyword arguments such as
    ``max_tokens`` are forwarded to :class:`llama_cpp.Llama`.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        from openai import OpenAI  # type: ignore

        model = kwargs.get("model", "gpt-3.5-turbo")
        client = OpenAI()
        response = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]

    if provider == "llama":
        from llama_cpp import Llama  # type: ignore

        model_path = kwargs.get("model_path")
        if model_path is None:
            raise ValueError("model_path must be provided for Llama provider")
        max_tokens = kwargs.get("max_tokens", 256)
        llm = Llama(model_path=model_path)
        output = llm(prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"].strip()

    raise ValueError(f"Unknown provider: {provider}")
