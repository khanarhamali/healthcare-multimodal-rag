from datetime import datetime, timezone
from fastapi import HTTPException
import psycopg
from src.rag.agents.graph import skin_care_graph, DB_URI
from src.services.storage import supabase, BUCKET_NAME

# Anti-hallucination guardrail: Maximum messages allowed per single thread (10 user + 10 assistant)
MAX_MESSAGES_PER_THREAD = 20


def verify_thread_capacity(user_id: str, thread_id: str) -> None:
    """Enforces the 20-message limit per thread to prevent context bloat and hallucination."""
    unique_langgraph_thread = f"{user_id}:{thread_id}"
    config = {"configurable": {"thread_id": unique_langgraph_thread}}
    state = skin_care_graph.get_state(config)

    if state and state.values:
        existing_messages = state.values.get("messages", [])
        if len(existing_messages) >= MAX_MESSAGES_PER_THREAD:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Thread limit reached ({MAX_MESSAGES_PER_THREAD} messages). "
                    "Please start a new consultation to maintain clinical accuracy."
                ),
            )


def upsert_thread_metadata(user_id: str, thread_id: str, is_new: bool, user_query: str | None) -> None:
    """Creates a new thread record or updates the timestamp of an existing one."""
    if is_new:
        raw_title = user_query if user_query else "Skin Image Analysis"
        thread_title = raw_title[:45] + ("..." if len(raw_title) > 45 else "")
        supabase.table("chat_threads").insert({
            "user_id": user_id,
            "thread_id": thread_id,
            "title": thread_title,
        }).execute()
    else:
        supabase.table("chat_threads").update({
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("thread_id", thread_id).execute()


def list_user_threads(user_id: str) -> list[dict]:
    """Fetches all saved consultation threads for a specific patient."""
    res = (
        supabase.table("chat_threads")
        .select("thread_id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return res.data or []


def fetch_thread_history(user_id: str, thread_id: str) -> dict:
    """Retrieves saved messages and capacity metadata for resuming a conversation."""
    unique_langgraph_thread = f"{user_id}:{thread_id}"
    config = {"configurable": {"thread_id": unique_langgraph_thread}}
    state = skin_care_graph.get_state(config)

    if not state or not state.values:
        return {
            "status": "success",
            "user_id": user_id,
            "thread_id": thread_id,
            "last_detected_condition": "None",
            "messages": [],
            "message_count": 0,
            "max_messages": MAX_MESSAGES_PER_THREAD,
        }

    raw_messages = state.values.get("messages", [])
    formatted_messages = [
        {
            "role": "user" if msg.type == "human" else "assistant",
            "content": msg.content,
        }
        for msg in raw_messages
    ]

    return {
        "status": "success",
        "user_id": user_id,
        "thread_id": thread_id,
        "last_detected_condition": state.values.get("detected_condition", "None"),
        "messages": formatted_messages,
        "message_count": len(raw_messages),
        "max_messages": MAX_MESSAGES_PER_THREAD,
    }


def purge_thread_hipaa(user_id: str, thread_id: str) -> None:
    """
    HIPAA Right-to-Erasure Deletion:
    1. Removes thread metadata from 'chat_threads'.
    2. Permanently purges LangGraph checkpoints from PostgreSQL.
    3. Deletes all patient images uploaded in this thread from Supabase Storage.
    """
    unique_langgraph_thread = f"{user_id}:{thread_id}"

    # 1. Remove metadata row
    supabase.table("chat_threads").delete().eq("user_id", user_id).eq("thread_id", thread_id).execute()

    # 2. Permanently delete LangGraph memory checkpoints from PostgreSQL
    with psycopg.connect(DB_URI, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (unique_langgraph_thread,))
            cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s;", (unique_langgraph_thread,))
            cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (unique_langgraph_thread,))

    # 3. Remove uploaded patient images from Supabase Storage
    folder_path = f"{user_id}/{thread_id}"
    files = supabase.storage.from_(BUCKET_NAME).list(folder_path)
    if files:
        file_paths = [f"{folder_path}/{f['name']}" for f in files if f.get("name")]
        if file_paths:
            supabase.storage.from_(BUCKET_NAME).remove(file_paths)
