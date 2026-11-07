from langgraph.graph import StateGraph, START, END
from src.rag.agents.state import AgentState
from src.rag.agents.nodes import vision_node, retrieval_node, generator_node

# Graph Wiring Configuration
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

    # --- Verification 3: Case 3 (Both Image & Text) ---
    print("--- Testing Case 3: Image + Text ---")
    case_3_output = skin_care_graph.invoke({"image_path": SAMPLE_IMAGE, "user_query": SAMPLE_TEXT})
    print(f"Condition: {case_3_output['detected_condition']}\nAnswer: {case_3_output['final_response'][:150]}...\n")
