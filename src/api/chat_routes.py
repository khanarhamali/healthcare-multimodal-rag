import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from src.services.thread_service import verify_thread_capacity
from src.services.chat_service import (
    upload_patient_image,
    run_standard_chat,
    stream_chat_events,
)

chat_router = APIRouter(tags=["Chat & Streaming"])


@chat_router.post("/chat")
async def chat_endpoint(
    user_id: str = Form("default_user"),
    thread_id: Optional[str] = Form(None),
    user_query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """Standard non-streaming multimodal chat endpoint."""
    if not user_query and not image:
        raise HTTPException(
            status_code=400,
            detail="Invalid request payload. Please supply text, an image, or both.",
        )

    is_new_thread = not bool(thread_id and thread_id.strip())
    active_thread_id = str(uuid.uuid4()) if is_new_thread else thread_id.strip()

    if not is_new_thread:
        verify_thread_capacity(user_id, active_thread_id)

    try:
        image_url = await upload_patient_image(image, user_id, active_thread_id) if image else None
        return run_standard_chat(user_id, active_thread_id, is_new_thread, user_query, image_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal agent server error: {str(e)}")


@chat_router.post("/chat/stream")
async def chat_stream_endpoint(
    user_id: str = Form(...),
    thread_id: Optional[str] = Form(None),
    user_query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
):
    """SSE streaming multimodal chat endpoint with 20-message anti-hallucination guardrail."""
    if not user_query and not image:
        raise HTTPException(
            status_code=400,
            detail="Invalid request. Please provide symptoms, a skin image, or both.",
        )

    is_new_thread = not bool(thread_id and thread_id.strip())
    active_thread_id = str(uuid.uuid4()) if is_new_thread else thread_id.strip()

    if not is_new_thread:
        verify_thread_capacity(user_id, active_thread_id)

    image_url = await upload_patient_image(image, user_id, active_thread_id) if image else None

    return StreamingResponse(
        stream_chat_events(user_id, active_thread_id, is_new_thread, user_query, image_url),
        media_type="text/event-stream",
    )
