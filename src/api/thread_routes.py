from fastapi import APIRouter, HTTPException
from src.services.thread_service import (
    list_user_threads,
    fetch_thread_history,
    purge_thread_hipaa,
)

thread_router = APIRouter(tags=["Threads & HIPAA Privacy"])


@thread_router.get("/threads/{user_id}")
async def get_user_threads(user_id: str):
    """Retrieves all saved consultation threads for a specific patient."""
    try:
        threads = list_user_threads(user_id)
        return {"status": "success", "user_id": user_id, "threads": threads}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch threads: {str(e)}")


@thread_router.get("/history/{user_id}/{thread_id}")
async def get_thread_history(user_id: str, thread_id: str):
    """Loads saved messages and state from Supabase PostgresSaver to resume a conversation."""
    try:
        return fetch_thread_history(user_id, thread_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat history: {str(e)}")


@thread_router.delete("/threads/{user_id}/{thread_id}")
async def delete_thread_hipaa(user_id: str, thread_id: str):
    """Permanently erases thread metadata, PostgreSQL checkpoints, and Supabase Storage images."""
    try:
        purge_thread_hipaa(user_id, thread_id)
        return {
            "status": "success",
            "message": "HIPAA Erasure Complete: Thread history, AI memory, and uploaded images have been permanently deleted.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HIPAA deletion failed: {str(e)}")
