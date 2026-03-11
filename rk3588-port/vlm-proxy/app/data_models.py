# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Ported from microservices/vlm-openvino-serving/src/utils/data_models.py
# Changes: removed OpenVINO-specific fields and imports; removed settings dependency;
# removed telemetry/internal models not needed by the proxy; renamed for OpenAI
# compatibility (ChatResponse, UsageInfo aliases added).

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MessageContentText(BaseModel):
    """Represents a text content part within a chat message."""

    type: str
    text: str


class MessageContentImageUrl(BaseModel):
    """Represents an image_url content part within a chat message."""

    type: str
    image_url: Dict[str, str]


class MessageContentVideo(BaseModel):
    """Represents a video (list of frame URLs) content part within a chat message."""

    type: str
    video: List[str]


class MessageContentVideoUrl(BaseModel):
    """Represents a video_url content part within a chat message."""

    type: str
    video_url: Dict[str, str]
    max_pixels: Optional[Union[int, str]] = None
    fps: Optional[float] = None


# Alias used by proxy.py for content part type unions
MessageContentPart = Union[
    str,
    MessageContentText,
    MessageContentImageUrl,
    MessageContentVideo,
    MessageContentVideoUrl,
]


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str
    content: Union[str, List[MessageContentPart]]


# Keep source name for backward compat
Message = ChatMessage


class UsageInfo(BaseModel):
    """Token usage statistics returned by llama-server (OpenAI-compatible)."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatRequest(BaseModel):
    """
    OpenAI-compatible chat completions request.

    All sampling parameters are passed through unchanged to llama-server.
    """

    messages: List[ChatMessage] = Field(...)
    model: str = Field(..., json_schema_extra={"example": "qwen2.5-vl"})
    repetition_penalty: Optional[float] = Field(None, json_schema_extra={"example": 1.15})
    presence_penalty: Optional[float] = Field(None, json_schema_extra={"example": 0.0})
    frequency_penalty: Optional[float] = Field(None, json_schema_extra={"example": 0.0})
    max_completion_tokens: Optional[int] = Field(None, json_schema_extra={"example": 1024})
    temperature: Optional[float] = Field(None, json_schema_extra={"example": 0.3})
    top_p: Optional[float] = Field(None, json_schema_extra={"example": 0.9})
    stream: Optional[bool] = Field(False, json_schema_extra={"example": True})
    top_k: Optional[int] = Field(None, json_schema_extra={"example": 40})
    do_sample: Optional[bool] = Field(None, json_schema_extra={"example": True})
    seed: Optional[int] = Field(None, json_schema_extra={"example": 42})


class ChatCompletionDelta(BaseModel):
    """Incremental delta for streaming chat completion chunks."""

    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """A single completion choice in a non-streaming response."""

    index: int
    message: ChatCompletionDelta
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response (non-streaming)."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None


class ChatCompletionStreamingChoice(BaseModel):
    """A single streaming choice delta."""

    index: int
    delta: ChatCompletionDelta
    finish_reason: Optional[str] = None
    usage: Optional[UsageInfo] = None


class ChatStreamingResponse(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    system_fingerprint: Optional[str] = None
    choices: List[ChatCompletionStreamingChoice]
