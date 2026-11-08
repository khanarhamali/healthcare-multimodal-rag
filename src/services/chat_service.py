import json
import uuid
from typing import AsyncGenerator
from fastapi import UploadFile, HTTPException
from langchain_core.messages import AIMessageChunk
from src.rag.agents.graph import skin_care_graph
from src.services.storage import supabase, BUCKET_NAME
from src.services.thread_service import (
    MAX_MESSAGES_PER_THREAD,
    upsert_thread_metadata,
)


async def upload_patient_image(image: UploadFile, user_id: str, thread_id: str) -> str:
    """Validates and uploads patient skin image to Supabase Storage."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File upload violation. Only valid image formats are permitted.",
        )

    file_extension = image.filename.split(".")[-1] if image.filename else "jpg"
    unique_filename = f"{user_id}/{thread_id}/{uuid.uuid4()}.{file_extension}"
    file_bytes = await image.read()

    supabase.storage.from_(BUCKET_NAME).upload(
        path=unique_filename,
        file=file_bytes,
        file_options={"content-type": image.content_type},
    )
    return supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)


def build_graph_payload(user_query: str | None, image_url: str | None) -> dict:
    """Constructs the LangGraph input dictionary with formatted message history."""
    user_msg_content = user_query if user_query else "[Uploaded Skin Image]"
    if image_url and user_query:
        user_msg_content = f"{user_query}\n[Image: {image_url}]"
    elif image_url:
        user_msg_content = f"[Image: {image_url}]"

    return {
        "messages": [("user", user_msg_content)],
        "user_query": user_query if user_query else None,
        "image_path": image_url,
    }


def run_standard_chat(
    user_id: str,
    thread_id: str,
    is_new_thread: bool,
    user_query: str | None,
    image_url: str | None,
) -> dict:
    """Executes LangGraph synchronously and returns the full response payload."""
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    graph_inputs = build_graph_payload(user_query, image_url)

    graph_output = skin_care_graph.invoke(graph_inputs, config=config)
    upsert_thread_metadata(user_id, thread_id, is_new_thread, user_query)

    return {
        "status": "success",
        "user_id": user_id,
        "thread_id": thread_id,
        "is_new_thread": is_new_thread,
        "detected_condition": graph_output.get("detected_condition", "None"),
        "processed_query": graph_output.get("user_query", ""),
        "uploaded_image_url": image_url,
        "response": graph_output.get("final_response", "Unable to formulate a response."),
    }


async def stream_chat_events(
    user_id: str,
    thread_id: str,
    is_new_thread: bool,
    user_query: str | None,
    image_url: str | None,
) -> AsyncGenerator[str, None]:
    """Streams real-time OpenAI tokens and vision predictions via SSE without waiting."""
    config = {"configurable": {"thread_id": f"{user_id}:{thread_id}"}}
    graph_inputs = build_graph_payload(user_query, image_url)

    try:
        # 1. Emit initial session metadata immediately
        meta_payload = {
            "type": "meta",
            "thread_id": thread_id,
            "is_new_thread": is_new_thread,
            "uploaded_image_url": image_url,
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

        # 2. Stream both node state updates AND live OpenAI LLM tokens simultaneously
        for mode, chunk in skin_care_graph.stream(
            graph_inputs, config=config, stream_mode=["updates", "messages"]
        ):
            if mode == "updates":
                for node_name, node_output in chunk.items():
                    if node_name == "vision_analyzer":
                        condition = node_output.get("detected_condition", "None")
                        yield f"data: {json.dumps({'type': 'condition', 'detected_condition': condition})}\n\n"

            elif mode == "messages":
                msg_chunk, metadata = chunk
                if (
                    isinstance(msg_chunk, AIMessageChunk)
                    and msg_chunk.content
                    and metadata.get("langgraph_node") == "response_generator"
                ):
                    yield f"data: {json.dumps({'type': 'token', 'content': msg_chunk.content})}\n\n"

        # 3. Persist thread metadata in Supabase
        upsert_thread_metadata(user_id, thread_id, is_new_thread, user_query)

        # 4. Emit completion event
        state = skin_care_graph.get_state(config)
        msg_count = len(state.values.get("messages", [])) if state and state.values else 0
        yield f"data: {json.dumps({'type': 'done', 'message_count': msg_count, 'max_messages': MAX_MESSAGES_PER_THREAD})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"
