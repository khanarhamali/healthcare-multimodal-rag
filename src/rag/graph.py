"""
src/rag/graph.py — Robust stateful routing graph handling all 3 user input cases.
"""

from typing import TypedDict, List, Optional
import os
from langgraph.graph import StateGraph, START, END

# Import your native components
from src.rag.tools import analyze_skin_image
from src.rag.test_retriever import build_retriever
from src.rag.test_generator import generate

# 1. State definition
class AgentState(TypedDict):
    user_query: Optional[str]
    image_path: Optional[str]
    detected_condition: Optional[str]
    retrieved_context: Optional[List[str]]
    final_response: Optional[str]

# Initialize the vector store retriever instance globally
retriever_instance = build_retriever()


# 2. Node A: Robust Vision Analyzer
def vision_node(state: AgentState) -> dict:
    """Handles cases with or without input image paths safely."""
    img_path = state.get("image_path")
    
    if not img_path or not os.path.exists(img_path):
        print("Graph Node [Vision]: Case 2 detected (Text Only). Skipping tool.")
        return {"detected_condition": "None"}
        
    print(f"Graph Node [Vision]: Case 1 or 3 detected. Running PyTorch tool on {os.path.basename(img_path)}...")
    condition_label = analyze_skin_image.invoke({"image_path": img_path})
    return {"detected_condition": condition_label}


# 3. Node B: Context Extractor with Default Query Fallback
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


# 4. Node C: Generator
def generator_node(state: AgentState) -> dict:
    """Generates the final response based on processed state variables."""
    condition = state.get("detected_condition", "None")
    query = state.get("user_query", "")
    chunks = state.get("retrieved_context", [])
    
    enriched_payload = f"Skin Condition: {condition}. User Query: {query}"
    
    print("Graph Node [Generator]: Producing grounded answer...")
    answer = generate(query=enriched_payload, context=chunks)
    return {"final_response": answer}


# 5. Graph Wiring Configuration
builder = StateGraph(AgentState)

builder.add_node("vision_analyzer", vision_node)
builder.add_node("context_retriever", retrieval_node)
builder.add_node("response_generator", generator_node)

builder.add_edge(START, "vision_analyzer")
builder.add_edge("vision_analyzer", "context_retriever")
builder.add_edge("context_retriever", "response_generator")
builder.add_edge("response_generator", END)

skin_care_graph = builder.compile()


# --- Local Testing Multi-Case Matrix Verification ---
if __name__ == "__main__":
    print("\n================== LAUNCHING LANGGRAPH RUNTIME TEST ==================\n")
    
    # Verification Configuration
    SAMPLE_IMAGE = "E:\\rag-eval\\cystic acne.jpg"
    SAMPLE_TEXT = "Is it fine to clean my face with warm water daily?"

    # --- Verification 1: Case 1 (Image Only) ---
    print("\n--- Testing Case 1: Image Only ---")
    case_1_output = skin_care_graph.invoke({"image_path": SAMPLE_IMAGE, "user_query": None})
    print(f"Condition: {case_1_output['detected_condition']}\nAnswer: {case_1_output['final_response'][:150]}...\n")

    # --- Verification 2: Case 2 (Text Only) ---
    print("--- Testing Case 2: Text Only ---")
    case_2_output = skin_care_graph.invoke({"image_path": None, "user_query": SAMPLE_TEXT})
    print(f"Condition: {case_2_output['detected_condition']}\nAnswer: {case_2_output['final_response'][:150]}...\n")

    # --- Verification 3: Case 3 (Dono Image & Text) ---
    print("--- Testing Case 3: Image + Text ---")
    case_3_output = skin_care_graph.invoke({"image_path": SAMPLE_IMAGE, "user_query": SAMPLE_TEXT})
    print(f"Condition: {case_3_output['detected_condition']}\nAnswer: {case_3_output['final_response'][:150]}...\n")
