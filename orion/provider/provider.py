import os
import asyncio
import litellm
from dataclasses import dataclass, field

MODELS = {
    "fast":     "nvidia_nim/meta/llama-3.1-8b-instruct",
    "coder":    "nvidia_nim/qwen/qwen2.5-coder-32b-instruct",
    "balanced": "nvidia_nim/meta/llama-3.3-70b-instruct",
    "heavy":    "nvidia_nim/deepseek/deepseek-v3",
}
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

class ProviderError(Exception):
    pass

class Provider:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        os.environ["NVIDIA_NIM_API_KEY"] = self.api_key
        # Token tracking — reset at start of each pipeline run
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def reset_token_counts(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def complete(
        self,
        tier: str,
        messages: list,
        stream: bool = False,
        on_token_delta = None,
    ) -> str:
        model = MODELS.get(tier, MODELS["fast"])
        kwargs = dict(
            model=model,
            messages=messages,
            api_key=self.api_key,
            api_base=NIM_BASE_URL,
            stream=stream,
        )
        try:
            return await self._call(kwargs, stream, on_token_delta)
        except ProviderError as e:
            # Retry once on rate limit
            if "rate_limit" in str(e).lower() or "529" in str(e):
                await asyncio.sleep(2)
                try:
                    return await self._call(kwargs, stream, on_token_delta)
                except Exception as e2:
                    raise ProviderError(str(e2))
            raise

    async def _call(self, kwargs, stream, on_token_delta) -> str:
        try:
            if stream:
                response = await litellm.acompletion(**kwargs)
                full = ""
                async for chunk in response:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        full += delta
                        if on_token_delta:
                            on_token_delta(delta)
                return full
            else:
                response = await litellm.acompletion(**kwargs)
                # Track tokens
                if hasattr(response, "usage") and response.usage:
                    self.total_prompt_tokens += response.usage.prompt_tokens or 0
                    self.total_completion_tokens += response.usage.completion_tokens or 0
                return response.choices[0].message.content or ""
        except Exception as e:
            raise ProviderError(str(e))
