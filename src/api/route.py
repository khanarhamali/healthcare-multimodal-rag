import os
import sys
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..")) if "src" in current_dir else current_dir
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.rag.agents.graph import skin_care_graph
from src.services.storage import supabase, BUCKET_NAME

# Router initialization
router = APIRouter(prefix="/api", tags=["Skin Care Agent"])


@router.post("/chat")
async def chat_endpoint(
    user_id: str = Form("default_user"),          # Unique identifier for the user
    thread_id: Optional[str] = Form(None),        # Leave empty for a new chat; pass existing ID to resume
    user_query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Supabase Storage and PostgreSQL Checkpointer integrated multimodal chat endpoint.
    """
    # Guardrail check: Ensure at least one input parameter is populated
    if not user_query and not image:
        raise HTTPException(
            status_code=400, 
            detail="Invalid request payload. Please supply either a text message, an image file, or both."
        )

    is_new_thread = False
    if not thread_id or not thread_id.strip():
        thread_id = str(uuid.uuid4())
        is_new_thread = True

    supabase_image_url = None

    try:
        # Step 1: Process and upload image to Supabase Storage if provided
        if image is not None:
            if not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, 
                    detail="File upload violation. The submitted document must be a valid image format."
                )
            
            file_extension = image.filename.split(".")[-1]
            # Organize uploaded images by user_id and thread_id
            unique_filename = f"{user_id}/{thread_id}/{uuid.uuid4()}.{file_extension}"
            file_bytes = await image.read()
            
            supabase.storage.from_(BUCKET_NAME).upload(
                path=unique_filename,
                file=file_bytes,
                file_options={"content-type": image.content_type}
            )
            
            supabase_image_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)

        # Step 2: Configure LangGraph thread isolation and prepare input state
        unique_langgraph_thread = f"{user_id}:{thread_id}"
        config = {"configurable": {"thread_id": unique_langgraph_thread}}

        # Format user message for persistent conversation history
        user_msg_content = user_query if user_query else "[Uploaded Skin Image]"
        if supabase_image_url and user_query:
            user_msg_content = f"{user_query}\n[Image: {supabase_image_url}]"
        elif supabase_image_url:
            user_msg_content = f"[Image: {supabase_image_url}]"

        graph_inputs = {
            "messages": [("user", user_msg_content)],
            "user_query": user_query if user_query else None,
            "image_path": supabase_image_url
        }

        print(f"\nExecuting LangGraph for user='{user_id}', thread='{thread_id}'...")
        # Step 3: Run runtime execution cycle through compiled LangGraph orchestrator
        graph_output = skin_care_graph.invoke(graph_inputs, config=config)

        # Step 4: Save or update thread metadata in Supabase 'chat_threads' table
        if is_new_thread:
            raw_title = user_query if user_query else "Skin Image Analysis"
            thread_title = raw_title[:45] + ("..." if len(raw_title) > 45 else "")
            supabase.table("chat_threads").insert({
                "user_id": user_id,
                "thread_id": thread_id,
                "title": thread_title
            }).execute()
        else:
            supabase.table("chat_threads").update({
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("thread_id", thread_id).execute()

        # Step 5: Construct unified standard response format
        return {
            "status": "success",
            "user_id": user_id,
            "thread_id": thread_id,
            "is_new_thread": is_new_thread,
            "detected_condition": graph_output.get("detected_condition", "None"),
            "processed_query": graph_output.get("user_query", ""),
            "uploaded_image_url": supabase_image_url,
            "response": graph_output.get("final_response", "Unable to formulate a response.")
        }

    except Exception as e:
        print(f"Server exception triggered inside chat endpoint routine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal agent server error: {str(e)}")


@router.get("/threads/{user_id}")
async def get_user_threads(user_id: str):
    """Retrieves all saved conversation threads for a specific user."""
    try:
        res = (
            supabase.table("chat_threads")
            .select("thread_id, title, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .execute()
        )
        return {"status": "success", "user_id": user_id, "threads": res.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch threads: {str(e)}")


@router.get("/history/{user_id}/{thread_id}")
async def get_thread_history(user_id: str, thread_id: str):
    """Loads saved messages and state from Supabase PostgresSaver to resume a conversation."""
    try:
        unique_langgraph_thread = f"{user_id}:{thread_id}"
        config = {"configurable": {"thread_id": unique_langgraph_thread}}
        
        state = skin_care_graph.get_state(config)
        if not state or not state.values:
            return {"status": "success", "thread_id": thread_id, "messages": []}

        formatted_messages = [
            {
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content
            }
            for msg in state.values.get("messages", [])
        ]

        return {
            "status": "success",
            "user_id": user_id,
            "thread_id": thread_id,
            "last_detected_condition": state.values.get("detected_condition", "None"),
            "messages": formatted_messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chat history: {str(e)}")


@router.delete("/threads/{user_id}/{thread_id}")
async def delete_thread(user_id: str, thread_id: str):
    """Deletes a conversation thread from the user's thread list."""
    try:
        supabase.table("chat_threads").delete().eq("user_id", user_id).eq("thread_id", thread_id).execute()
        return {"status": "success", "message": "Thread deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete thread: {str(e)}")
