import os
from src.rag.agents.state import AgentState
from src.rag.tools.tools import analyze_skin_image
from src.rag.components.retriever import build_retriever
from src.rag.components.generator import generate

# Initialize the vector store retriever instance globally
retriever_instance = build_retriever()


def vision_node(state: AgentState) -> dict:
    """Handles cases with or without input image paths/URLs safely."""
    img_path = state.get("image_path")
    
    is_url = isinstance(img_path, str) and img_path.startswith(("http://", "https://"))
    is_local_file = isinstance(img_path, str) and os.path.exists(img_path)

    if not img_path or (not is_url and not is_local_file):
        print("Graph Node [Vision]: Case 2 detected (Text Only). Skipping tool.")
        return {"detected_condition": "None"}
        
    print(f"Graph Node [Vision]: Case 1 or 3 detected. Running PyTorch tool on {os.path.basename(img_path)}...")
    condition_label = analyze_skin_image.invoke({"image_path": img_path})
    return {"detected_condition": condition_label}


def retrieval_node(state: AgentState) -> dict:
    """Safely extracts context even if the user provided no text question."""
    condition = state.get("detected_condition", "None")
    original_query = state.get("user_query", "").strip() if state.get("user_query") else ""
    
    # Handle Case 1: If user only uploaded an image, inject a default educational query
    if not original_query and condition != "None":
        print("Graph Node [Retriever]: Case 1 detected (Image Only). Substituting default query.")
        original_query = f"What is the recommended treatment and general daily care routine for {condition}?"
    
    # Enrich look-up payload
    enriched_payload = f"Skin Condition: {condition}. User Query: {original_query}"
    print(f"Graph Node [Retriever]: Querying database with: '{enriched_payload}'")
    
    docs = retriever_instance.invoke(enriched_payload)
    string_chunks = [doc.page_content for doc in docs]
    
    # Update the query inside state so the generator knows what query we fulfilled
    return {"retrieved_context": string_chunks, "user_query": original_query}


def generator_node(state: AgentState) -> dict:
    """Generates the final response based on processed state variables and saves it to history."""
    condition = state.get("detected_condition", "None")
    query = state.get("user_query", "")
    chunks = state.get("retrieved_context", [])
    
    enriched_payload = f"Skin Condition: {condition}. User Query: {query}"
    
    print("Graph Node [Generator]: Producing grounded answer...")
    answer = generate(query=enriched_payload, context=chunks)
    return {
        "final_response": answer,
        "messages": [("assistant", answer)],
    }
