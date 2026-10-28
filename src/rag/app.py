"""
app.py — FastAPI endpoint serving the compiled LangGraph skin care agent.
Handles:
  1) Image Only
  2) Text Only
  3) Image + Text
"""

import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Setup global path checking to keep imports functional
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..")) if "src" in current_dir else current_dir
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import your compiled state graph orchestrator from graph.py
from src.rag.graph import skin_care_graph

app = FastAPI(
    title="Skin Care Multimodal RAG API",
    description="Unified API interface for PyTorch Vision Analysis + FAISS Context Retrieval + OpenAI Generation via LangGraph.",
    version="1.0.0"
)

# Enable CORS so frontend frameworks (React, Streamlit, or local HTML) can test smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = os.path.join(root_dir, "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/api/chat")
async def chat_endpoint(
    user_query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Unified entry endpoint handling all 3 multimodal chat variations.
    """
    # Guardrail check: Ensure at least one input parameter is populated
    if not user_query and not image:
        raise HTTPException(
            status_code=400, 
            detail="Invalid request payload. Please supply either a text message, an image file, or both."
        )

    local_image_path = None

    try:
        # Step 1: Process and store the incoming image stream locally if available
        if image is not None:
            if not image.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, 
                    detail="File upload violation. The submitted document must be a valid image format."
                )
            
            local_image_path = os.path.join(TEMP_DIR, image.filename)
            with open(local_image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

        # Step 2: Structure payload variables matching your compiled AgentState layout
        graph_inputs = {
            "user_query": user_query if user_query else None,
            "image_path": local_image_path
        }

        print(f"\nExposing endpoint request execution chain to LangGraph machine...")
        # Step 3: Run runtime execution cycle through compiled LangGraph orchestrator
        graph_output = skin_care_graph.invoke(graph_inputs)

        # Step 4: Storage house cleaning (Remove temp image asset to save disk space)
        if local_image_path and os.path.exists(local_image_path):
            os.remove(local_image_path)

        # Step 5: Construct unified standard response format
        return {
            "status": "success",
            "detected_condition": graph_output.get("detected_condition", "None"),
            "processed_query": graph_output.get("user_query", ""),
            "response": graph_output.get("final_response", "Unable to formulate a response.")
        }

    except Exception as e:
        # Emergency file cleanup in case runtime pipeline drops mid-flight
        if local_image_path and os.path.exists(local_image_path):
            os.remove(local_image_path)
            
        print(f"Server exception triggered inside chat endpoint routine: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal agent server error: {str(e)}")


@app.get("/health")
def health_check():
    """Confirms operational integrity of the server application layer."""
    return {"status": "healthy", "service": "Skin Care RAG Pipeline"}
