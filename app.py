from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.api.route import router as skin_care_router 

app = FastAPI(
    title="Skin Care Multimodal RAG API",
    description="Unified API interface for PyTorch Vision Analysis + FAISS Context Retrieval + OpenAI Generation via LangGraph.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(skin_care_router)

@app.get("/health")
def health_check():
    """Confirms operational integrity of the server application layer."""
    return {"status": "healthy", "service": "Skin Care RAG Pipeline"}


if __name__ == "__main__":
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
