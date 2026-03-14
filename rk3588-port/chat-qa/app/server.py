# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx

from .chain import process_chunks

app = FastAPI(title="Chat Question and Answer", root_path="/v1/chatqna")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=os.getenv("CORS_ALLOW_METHODS", "*").split(","),
    allow_headers=os.getenv("CORS_ALLOW_HEADERS", "*").split(","),
)


# ---------------------------------------------------------------------------
# Health-check helpers
# ---------------------------------------------------------------------------

async def check_health(url: str, server_type: str):
    """
    Check remote model server readiness by requesting its health endpoint.
    
    Parameters:
        url (str): The full URL of the server's health endpoint to probe.
        server_type (str): Human-readable name for the server used in returned details and error messages.
    
    Returns:
        dict: `{'status': 'healthy', 'details': '<server_type> is ready to serve'}` when the endpoint responds with HTTP 200.
    
    Raises:
        HTTPException: 503 when the server responds with a non-200 status or when the request fails.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5.0)
            if response.status_code == 200:
                return {"status": "healthy", "details": f"{server_type} is ready to serve"}
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"{server_type} is not ready to accept connections, "
                           "please try after a few minutes",
                )
        except httpx.RequestError as err:
            raise HTTPException(
                status_code=503,
                detail=f"{server_type} is not ready to accept connections, "
                       "please try after a few minutes",
            ) from err


async def check_server_health(host: str, server_type: str):
    """
    Selects the appropriate health-check endpoint for a model server host and delegates the check.
    
    Parameters:
        host (str): Hostname (optionally with port) of the model server.
        server_type (str): Label identifying the server type used in the returned details.
    
    Returns:
        dict: Health status result as returned by the underlying check (e.g., `{"status": "healthy", "details": "<server_type> is ready to serve"}`).
    """
    if host.startswith(("vllm", "text", "tei", "llama")):
        return await check_health(f"http://{host}/health", server_type)
    elif host.startswith(("ovms", "openvino")):
        return await check_health(f"http://{host}/v2/health/ready", server_type)
    else:
        # For generic hosts (e.g. localhost) try /health as a best-effort check.
        return await check_health(f"http://{host}/health", server_type)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class QuestionRequest(BaseModel):
    conversation_messages: List[Message]
    max_tokens: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def redirect_root_to_docs():
    """
    Redirects incoming requests to the API documentation at /docs.
    
    Returns:
        RedirectResponse: HTTP redirect response pointing to "/docs".
    """
    return RedirectResponse("/docs")


@app.get("/health")
async def health_check():
    """
    Aggregate health status of the configured LLM and embedding model servers.
    
    Checks the environment variables LLM_ENDPOINT_URL and EMBEDDING_ENDPOINT_URL, probes each server's health endpoint, and returns their health results.
    
    Returns:
        list[dict]: A list of health status objects for each server. Each object contains at least the keys `status` (e.g., "healthy") and `details` (human-readable message).
    
    Raises:
        HTTPException: 503 if either endpoint environment variable is missing or if any server reports an unhealthy state.
    """
    endpoint_url = os.getenv("LLM_ENDPOINT_URL")
    embedding_endpoint = os.getenv("EMBEDDING_ENDPOINT_URL")

    if not endpoint_url or not embedding_endpoint:
        raise HTTPException(
            status_code=503,
            detail="LLM_ENDPOINT_URL or EMBEDDING_ENDPOINT_URL is not set",
        )

    result = []
    model_host = endpoint_url.split("//")[-1].split("/")[0].lower()
    result.append(await check_server_health(model_host, "LLM model server"))

    embed_host = embedding_endpoint.split("//")[-1].split("/")[0].lower()
    result.append(await check_server_health(embed_host, "Embedding model server"))

    return result


@app.get("/model")
async def get_llm_model():
    """
    Return the configured LLM model name.
    
    Returns:
        dict: Response object with keys:
            - "status": always "success"
            - "llm_model": the configured LLM model name
    
    Raises:
        HTTPException: with status code 503 if the environment variable `LLM_MODEL_NAME` is not set.
    """
    llm_model = os.getenv("LLM_MODEL_NAME")
    if not llm_model:
        raise HTTPException(status_code=503, detail="LLM_MODEL_NAME is not set")
    return {"status": "success", "llm_model": llm_model}


@app.post("/chat", response_class=StreamingResponse)
async def query_chain(payload: QuestionRequest):
    """
    Handles POST requests to the /chat endpoint.

    Receives a conversation history and the current question, validates the input,
    and returns a streaming SSE response with the LLM-generated answer.

    Args:
        payload (QuestionRequest): Conversation history and max_tokens.

    Returns:
        StreamingResponse: SSE stream of LLM output chunks.

    Raises:
        HTTPException: 422 if question is empty; 500 for unexpected errors.
    """
    try:
        conversation_messages = payload.conversation_messages
        if not conversation_messages:
            raise HTTPException(
                status_code=422,
                detail="conversation_messages must contain at least one message",
            )
        question_text = conversation_messages[-1].content

        max_tokens = payload.max_tokens if payload.max_tokens else 512
        if max_tokens > 1024:
            raise HTTPException(
                status_code=422, detail="max tokens cannot be greater than 1024"
            )
        if not question_text or question_text == "":
            raise HTTPException(status_code=422, detail="Question is required")
        if len(question_text.strip()) == 0:
            raise HTTPException(
                status_code=422, detail="Question cannot be empty or whitespace only"
            )

        return StreamingResponse(
            process_chunks(conversation_messages, max_tokens),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
