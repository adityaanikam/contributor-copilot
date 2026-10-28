"""FastAPI app. Run with: uvicorn contributor_copilot.api:app"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from ollama import ResponseError
from pydantic import BaseModel, Field

from contributor_copilot.config import Settings
from contributor_copilot.service import AskResult, Copilot, build_copilot


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(copilot: Copilot | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Tests pass a ready Copilot; otherwise it is built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.copilot is None:
            app.state.copilot = build_copilot(settings or Settings.from_env())
        yield

    app = FastAPI(title="Contributor Copilot", lifespan=lifespan)
    app.state.copilot = copilot

    @app.post("/ask")
    def ask(request: AskRequest) -> AskResult:
        try:
            return app.state.copilot.ask(request.question.strip())
        except (httpx.ConnectError, ConnectionError) as error:
            raise HTTPException(503, f"LLM backend unreachable. Is Ollama running? ({error})")
        except ResponseError as error:
            raise HTTPException(502, f"LLM backend error: {error}")

    @app.get("/health")
    def health() -> dict:
        retriever = app.state.copilot.retriever
        return {"status": "ok", "retrieval": retriever.mode, "chunks": len(retriever.bm25.chunks)}

    return app


app = create_app()
