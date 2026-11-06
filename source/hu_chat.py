from openai import OpenAI, AsyncOpenAI
import time
import asyncio
import threading
import re

DEFAULT_BASE_URL = "https://llm3-compute.cms.hu-berlin.de/v1/"

# Sampling temperature sent to the endpoint. Previously unset (the server applied
# its own default, ~1.0), which made extractions highly non-deterministic. Pin it
# low for more stable, reproducible runs.
DEFAULT_TEMPERATURE = 0.3
# Per-request timeout in seconds. The OpenAI client's default is 600 s: a judge call takes
# under 2 s, and an occasional hung request would otherwise block a whole batch for
# 10 minutes. Retries are left to the 10-attempt loops below (client max_retries=0).
REQUEST_TIMEOUT_S = 120
# Back-off between retries of a failed request.
RETRY_SLEEP_S = 10
# Cache for model names to avoid repeated lookups
_model_cache = {}
_sync_client_cache = {}
_async_client_cache = {}
_cache_lock = threading.Lock()


def _strip_thinking_content(text: str | None) -> str:
    """Remove model reasoning blocks from output.

    Handles three formats:
    - Complete blocks:  <think>...</think>answer
    - Dangling open:    <think>...answer  (no closing tag)
    - Dangling close:   Thinking Process:...</think>\n\nanswer  (Qwen3.5 style)
    """
    if text is None:
        return ""

    cleaned = str(text)

    # Remove complete think blocks.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    lower = cleaned.lower()

    # Dangling opening <think> tag — take everything after </think> or after the tag itself.
    if "<think" in lower:
        close_idx = lower.rfind("</think>")
        if close_idx != -1:
            cleaned = cleaned[close_idx + len("</think>"):]
        else:
            tag_end = lower.find(">", lower.find("<think"))
            if tag_end != -1:
                cleaned = cleaned[tag_end + 1:]
    # Dangling closing </think> tag (Qwen3.5: "Thinking Process:...\n</think>\n\nanswer").
    elif "</think>" in lower:
        close_idx = lower.rfind("</think>")
        cleaned = cleaned[close_idx + len("</think>"):]

    return cleaned.strip()


def _get_sync_client(base_url: str) -> OpenAI:
    with _cache_lock:
        client = _sync_client_cache.get(base_url)
        if client is None:
            client = OpenAI(base_url=base_url, api_key="required-but-not-used")
            _sync_client_cache[base_url] = client
        return client


def _get_async_client(base_url: str) -> AsyncOpenAI:
    with _cache_lock:
        client = _async_client_cache.get(base_url)
        if client is None:
            client = AsyncOpenAI(base_url=base_url, api_key="required-but-not-used",
                     timeout=REQUEST_TIMEOUT_S, max_retries=0)
            _async_client_cache[base_url] = client
        return client


def _resolve_model(base_url: str, model: str | None = None) -> str:
    if model is not None:
        return model

    with _cache_lock:
        cached = _model_cache.get(base_url)
    if cached is not None:
        return cached

    sync_client = _get_sync_client(base_url)
    models = sync_client.models.list()
    model_to_use = models.data[0].id

    with _cache_lock:
        _model_cache[base_url] = model_to_use

    return model_to_use


def hu_instruct(prompt, args_base_url=None, model=None, temperature: float = DEFAULT_TEMPERATURE):
    """
    Send a prompt to an LLM endpoint (synchronous version).

    Args:
        prompt: The prompt to send
        args_base_url: The base URL of the LLM endpoint. Defaults to HU's llm3-compute.
        model: The model name to use. If None and using HU endpoint, auto-detects.
               Required when using non-HU endpoints (e.g., local vLLM).
        temperature: Sampling temperature (default DEFAULT_TEMPERATURE = 0.3).
    """
    attempts = 0
    base_url = args_base_url if args_base_url is not None else DEFAULT_BASE_URL

    while attempts < 10:
        try:
            client = _get_sync_client(base_url)
            model_to_use = _resolve_model(base_url, model)

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_to_use,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return _strip_thinking_content(chat_completion.choices[0].message.content)
        
        except Exception as e:
            attempts += 1
            print(f"Error in hu llm: {e}. Retrying... ({attempts}/10)")
            time.sleep(10)
    raise RuntimeError("hu_instruct failed after 10 consecutive retries")


async def hu_instruct_async(prompt, args_base_url=None, model=None, semaphore=None, max_tokens: int = 500,
                            temperature: float = DEFAULT_TEMPERATURE, seed: int | None = None):
    """
    Send a prompt to an LLM endpoint (asynchronous version).

    Args:
        prompt: The prompt to send
        args_base_url: The base URL of the LLM endpoint. Defaults to HU's llm3-compute.
        model: The model name to use. If None, will try to auto-detect.
        semaphore: Optional asyncio.Semaphore to limit concurrency.
        max_tokens: Maximum tokens to generate. Keep small for classification prompts.
        temperature: Sampling temperature (default DEFAULT_TEMPERATURE = 0.3).
        seed: Optional sampling seed forwarded to the endpoint. With a fixed seed and
              temperature, identical prompts return identical outputs; varying the seed
              across runs yields the run-to-run variance used for error bars.
    """
    base_url = args_base_url if args_base_url is not None else DEFAULT_BASE_URL
    client = _get_async_client(base_url)
    # The model lookup is a blocking request; run it in a thread so the first call does not
    # stall the event loop (later calls hit the cache).
    model_to_use = model if model is not None else await asyncio.to_thread(_resolve_model, base_url)

    async def _request():
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\n/no_think",
                }
            ],
            model=model_to_use,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return _strip_thinking_content(chat_completion.choices[0].message.content)

    # The semaphore is held for one request at a time, not during the back-off sleep, so
    # a failing request does not keep a concurrency slot idle for ten seconds.
    attempts = 0
    while True:
        try:
            if semaphore:
                async with semaphore:
                    return await _request()
            return await _request()
        except Exception as e:
            attempts += 1
            print(f"Error in hu llm async: {e}. Retrying... ({attempts}/10)")
            if attempts >= 10:
                raise RuntimeError("hu_instruct_async failed after 10 consecutive retries") from e
            await asyncio.sleep(RETRY_SLEEP_S)
