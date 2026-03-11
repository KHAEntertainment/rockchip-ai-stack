# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# VLM video-preprocessing proxy for the RK3588 port.
#
# Responsibilities:
#   - Forward text/image-only requests to llama-server unchanged (zero overhead).
#   - For requests that contain video content parts, extract frames with
#     extract_qwen_video_frames(), convert them to base64 PNG image_url parts,
#     then forward the rewritten request to llama-server.
#   - Pass SSE streaming responses through transparently without buffering.
#   - Expose a /health endpoint that surfaces both proxy and llama-server status.

import base64
import logging
import os
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import Field
from pydantic_settings import BaseSettings
from starlette.responses import StreamingResponse

from app.data_models import (
    ChatMessage,
    ChatRequest,
    MessageContentImageUrl,
    MessageContentPart,
    MessageContentVideo,
    MessageContentVideoUrl,
)
from app.utils import decode_and_save_video, extract_qwen_video_frames

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables / .env file."""

    llama_server_url: str = "http://localhost:8080"
    max_video_frames: int = 8
    port: int = 8082

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VLM Video-Preprocessing Proxy",
    description=(
        "Thin proxy that extracts video frames and rewrites video content "
        "parts as image_url parts before forwarding to llama-server."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pil_to_base64_png(image) -> str:
    """Encode a PIL image as a base64 PNG data-URI."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _has_video_content(messages: List[ChatMessage]) -> bool:
    """Return True if any message contains a video or video_url content part."""
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if isinstance(part, (MessageContentVideo, MessageContentVideoUrl)):
                return True
    return False


async def _expand_video_part(
    part: MessageContentPart,
    max_frames: int,
) -> List[MessageContentPart]:
    """Convert a single video content part into a list of image_url parts.

    For ``MessageContentVideoUrl`` the video is first saved to /tmp via
    ``decode_and_save_video`` when the URL is a base64 data-URI.  Then
    ``extract_qwen_video_frames`` is called to sample frames.  Each frame
    becomes an ``image_url`` content part carrying a base64 PNG data-URI.

    For ``MessageContentVideo`` (a list of frame URLs / base64 images) the
    frames are loaded directly.

    Returns:
        A list of ``MessageContentImageUrl`` parts (one per sampled frame),
        or the original part unchanged when it is not a video type.
    """
    if isinstance(part, MessageContentVideoUrl):
        url = part.video_url.get("url", "")
        if url.startswith("data:video/"):
            # Inline base64-encoded video — decode to a temp file first.
            local_path = decode_and_save_video(url)
            video_url = local_path
        elif url.startswith("http://") or url.startswith("https://"):
            # Remote video URL — pass through directly.
            video_url = url
        else:
            raise ValueError(
                f"Unsupported video URL: only 'data:video/' and 'http(s)://' "
                f"schemes are accepted, got {url[:80]!r}"
            )

        # Use decord / pyav to decode the video into numpy arrays
        try:
            from decord import VideoReader, cpu  # type: ignore

            vr = VideoReader(video_url.replace("file://", ""), ctx=cpu(0))
            total = len(vr)
            step = max(1, total // max_frames)
            indices = list(range(0, total, step))[:max_frames]
            raw_frames = vr.get_batch(indices).asnumpy()  # (N, H, W, C) uint8
        except ImportError:
            logger.warning("decord not available; attempting OpenCV fallback")
            import cv2  # type: ignore
            import numpy as np

            cap = cv2.VideoCapture(video_url.replace("file://", ""))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // max_frames)
            raw_frames_list = []
            for i in range(0, total, step):
                if len(raw_frames_list) >= max_frames:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    raw_frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            raw_frames = np.stack(raw_frames_list, axis=0) if raw_frames_list else np.empty((0,))

        import numpy as np
        from PIL import Image

        image_parts: List[MessageContentPart] = []
        for i in range(raw_frames.shape[0]):
            pil_img = Image.fromarray(raw_frames[i].astype(np.uint8))
            image_parts.append(
                MessageContentImageUrl(
                    type="image_url",
                    image_url={"url": _pil_to_base64_png(pil_img)},
                )
            )
        if not image_parts:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Video URL produced zero frames; cannot forward request.",
            )
        return image_parts

    if isinstance(part, MessageContentVideo):
        # part.video is a list of frame URLs or base64 images
        from app.utils import load_images

        images, _ = await load_images(part.video)
        frames = extract_qwen_video_frames(
            [__import__("numpy").stack(
                [__import__("numpy").array(img) for img in images], axis=0
            )],
            max_frames=max_frames,
        ) if images else []
        if not frames:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Video content produced zero frames; cannot forward request.",
            )
        return [
            MessageContentImageUrl(
                type="image_url",
                image_url={"url": _pil_to_base64_png(frame)},
            )
            for frame in frames
        ]

    return [part]


async def _rewrite_messages_for_video(
    messages: List[ChatMessage],
    max_frames: int,
) -> List[ChatMessage]:
    """Return a copy of *messages* with video parts replaced by image_url parts."""
    rewritten: List[ChatMessage] = []
    for message in messages:
        if not isinstance(message.content, list):
            rewritten.append(message)
            continue

        new_parts: List[MessageContentPart] = []
        for part in message.content:
            expanded = await _expand_video_part(part, max_frames)
            new_parts.extend(expanded)

        rewritten.append(ChatMessage(role=message.role, content=new_parts))
    return rewritten


def _build_forward_payload(
    raw_body: Dict[str, Any],
    rewritten_messages: Optional[List[ChatMessage]],
) -> Dict[str, Any]:
    """Build the forwarding payload from the raw request body.

    When *rewritten_messages* is None the raw body is forwarded as-is (no
    video rewriting was needed), preserving every field the upstream client
    sent.  When video frames were extracted, only the ``messages`` key is
    replaced so that all other original fields (vendor extensions, sampling
    params, etc.) are still forwarded unchanged.
    """
    if rewritten_messages is None:
        return dict(raw_body)
    payload = dict(raw_body)
    payload["messages"] = [m.model_dump(exclude_none=True) for m in rewritten_messages]
    return payload


async def _stream_response(
    url: str,
    payload: Dict[str, Any],
    upstream_headers: Dict[str, str],
) -> AsyncIterator[bytes]:
    """Yield raw SSE bytes from llama-server without buffering.

    The AsyncClient is created and owned inside this generator so that it
    stays alive for the full duration of the SSE stream, even after the
    calling endpoint handler has returned the StreamingResponse.
    """
    client = httpx.AsyncClient()
    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=upstream_headers,
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, chat_request: ChatRequest):
    """OpenAI-compatible chat completions endpoint.

    - Requests without video content are forwarded to llama-server unchanged.
    - Requests with video content have their video parts replaced with sampled
      image_url parts before forwarding.
    - SSE streaming is passed through transparently.
    """
    target_url = f"{settings.llama_server_url}/v1/chat/completions"

    # Preserve selected headers from the upstream client
    forward_headers: Dict[str, str] = {"Content-Type": "application/json"}
    if "authorization" in request.headers:
        forward_headers["Authorization"] = request.headers["authorization"]

    raw_body: Dict[str, Any] = await request.json()

    if _has_video_content(chat_request.messages):
        logger.info("Video content detected — extracting frames before forwarding")
        rewritten = await _rewrite_messages_for_video(
            chat_request.messages, settings.max_video_frames
        )
    else:
        rewritten = None  # forward raw body unchanged

    payload = _build_forward_payload(raw_body, rewritten)
    is_streaming = bool(payload.get("stream"))

    if is_streaming:
        # _stream_response owns its own AsyncClient for the full SSE lifetime.
        return StreamingResponse(
            _stream_response(target_url, payload, forward_headers),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                target_url,
                json=payload,
                headers=forward_headers,
                timeout=120.0,
            )
            response.raise_for_status()
            return JSONResponse(content=response.json(), status_code=response.status_code)
        except httpx.HTTPStatusError as exc:
            logger.error("llama-server returned error: %s", exc)
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            ) from exc
        except httpx.RequestError as exc:
            logger.error("Could not reach llama-server: %s", exc)
            raise HTTPException(
                status_code=503, detail=f"llama-server unreachable: {exc}"
            ) from exc


@app.get("/health")
async def health():
    """Health-check endpoint.

    Probes llama-server's /health route.  Returns HTTP 200 when the upstream
    is reachable or HTTP 503 with degraded status otherwise.
    """
    llama_health_url = f"{settings.llama_server_url}/health"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(llama_health_url, timeout=5.0)
            resp.raise_for_status()
        llama_status = "reachable"
        status_code = 200
    except Exception as exc:
        logger.warning("llama-server health check failed: %s", exc)
        llama_status = "unreachable"
        status_code = 503

    body: Dict[str, str] = {
        "status": "ok" if status_code == 200 else "degraded",
        "llama_server": llama_status,
        "proxy": "vlm-proxy",
    }
    return JSONResponse(content=body, status_code=status_code)


# ---------------------------------------------------------------------------
# Entry point (for direct execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.proxy:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )
