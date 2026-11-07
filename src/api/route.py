import os
import sys
import uuid
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
    user_query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Supabase Storage integrated multimodal chat endpoint using separate config layer.
    """
    # Guardrail check: Ensure at least one input parameter is populated
    if not user_query and not image:
        raise HTTPException(
            status_code=400, 
            detail="Invalid request payload. Please supply either a text message, an image file, or both."
        )

    supabase_image_url = None

    try:
        # Step 1: Process and upload image to Supabase if provided
        if image is not None:
            if not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, 
                    detail="File upload violation. The submitted document must be a valid image format."
                )
            
            # Unique file name generation using UUID
            file_extension = image.filename.split(".")[-1]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            
            # Read file contents into memory bytes
            file_bytes = await image.read()
            
            # Direct binary stream upload to Supabase Storage
            supabase.storage.from_(BUCKET_NAME).upload(
                path=unique_filename,
                file=file_bytes,
                file_options={"content-type": image.content_type}
            )
            
            # Generate public URL for the newly hosted image asset
            public_url_response = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)
            supabase_image_url = public_url_response

        # Step 2: Structure payload variables matching your compiled AgentState layout
        graph_inputs = {
            "user_query": user_query if user_query else None,
            "image_path": supabase_image_url 
        }

        print(f"\nExposing endpoint request execution chain to LangGraph machine...")
        # Step 3: Run runtime execution cycle through compiled LangGraph orchestrator
        graph_output = skin_care_graph.invoke(graph_inputs)

        # Step 4: Construct unified standard response format
        return {
            "status": "success",
            "detected_condition": graph_output.get("detected_condition", "None"),
            "processed_query": graph_output.get("user_query", ""),
            "uploaded_image_url": supabase_image_url,
            "response": graph_output.get("final_response", "Unable to formulate a response.")
        }

    except Exception as e:
        print(f"Server exception triggered inside chat endpoint routine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal agent server error: {str(e)}")
