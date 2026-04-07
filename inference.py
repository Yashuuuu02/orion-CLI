import os
import asyncio
import json
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Orion CLI - OpenEnv API")
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )

api_key = os.environ.get("NVIDIA_NIM_API_KEY", "")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "orion"
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict


class ResetRequest(BaseModel):
    difficulty: Optional[str] = None


class StepRequest(BaseModel):
    prompt: str


_state = {}


@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    return {"status": "ok", "service": "Orion CLI OpenEnv"}


@app.get("/health")
@limiter.limit("60/minute")
async def health(request: Request):
    return {"status": "healthy"}


@app.post("/v1/chat/completions")
@limiter.limit("10/minute")
async def chat_completions(request: Request):
    req = await request.json()
    if not api_key:
        raise HTTPException(status_code=401, detail="NVIDIA_NIM_API_KEY not configured")

    if not req.get("messages"):
        raise HTTPException(status_code=400, detail="No messages provided")

    user_message = req["messages"][-1]["content"]

    from env import get_env
    env = get_env(api_key)

    if "reset" not in _state:
        await env.reset()
        _state["reset"] = True

    try:
        result_state, reward, done = await env.step(user_message)

        response_content = json.dumps({
            "response": result_state.get("history", [])[-1].get("response", "") if result_state.get("history") else "",
            "reward": reward,
            "done": done,
            "total_reward": result_state.get("total_reward", 0),
            "steps": result_state.get("steps", 0),
            "iisg_rate": result_state.get("history", [])[-1].get("iisg_rate", 0) if result_state.get("history") else 0,
        })

        return ChatCompletionResponse(
            id=f"orion-{int(asyncio.get_event_loop().time() * 1000)}",
            created=int(asyncio.get_event_loop().time()),
            model=req.get("model", "orion"),
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_content),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": len(response_content),
                "total_tokens": len(response_content)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
@limiter.limit("20/minute")
async def reset(request: Request):
    req_json = await request.json()
    if not api_key:
        raise HTTPException(status_code=401, detail="NVIDIA_NIM_API_KEY not configured")

    from env import get_env
    env = get_env(api_key)
    difficulty = req_json.get("difficulty")
    result = await env.reset(difficulty)
    _state["reset"] = True
    _state["difficulty"] = difficulty
    return result


@app.post("/step")
@limiter.limit("20/minute")
async def step(request: Request):
    req_json = await request.json()
    if not api_key:
        raise HTTPException(status_code=401, detail="NVIDIA_NIM_API_KEY not configured")

    if "reset" not in _state:
        raise HTTPException(status_code=400, detail="Call /reset first")

    from env import get_env
    env = get_env(api_key)
    prompt = req_json.get("prompt", "")
    result_state, reward, done = await env.step(prompt)

    return {
        "state": result_state,
        "reward": reward,
        "done": done
    }


@app.get("/state")
@limiter.limit("30/minute")
async def get_state(request: Request):
    if "reset" not in _state:
        raise HTTPException(status_code=400, detail="Call /reset first")

    from env import get_env
    env = get_env(api_key)
    return await env.state()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)