import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from src.rag.agents.state import AgentState
from src.rag.agents.nodes import vision_node, retrieval_node, generator_node

load_dotenv()

# Graph Wiring Configuration
builder = StateGraph(AgentState)

builder.add_node("vision_analyzer", vision_node)
builder.add_node("context_retriever", retrieval_node)
builder.add_node("response_generator", generator_node)

builder.add_edge(START, "vision_analyzer")
builder.add_edge("vision_analyzer", "context_retriever")
builder.add_edge("context_retriever", "response_generator")
builder.add_edge("response_generator", END)

# --- Supabase PostgreSQL Checkpointer Setup (Auto-Reconnect & Pooler Safe) ---
DB_URI = os.getenv("SUPABASE_DB_URL")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": None,  # Completely disables prepared statements for Supabase Transaction Pooler (port 6543)
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}

pool = ConnectionPool(
    conninfo=DB_URI,
    min_size=1,
    max_size=20,
    max_lifetime=300,                       # Recycle connections every 5 minutes before Supabase closes idle sockets
    kwargs=connection_kwargs,
)

# Initialize PostgresSaver and auto-create LangGraph checkpoint tables in Supabase
memory = PostgresSaver(pool)
memory.setup()

# Compile graph with persistent memory checkpointer
skin_care_graph = builder.compile(checkpointer=memory)


# --- Local Testing Multi-Case Matrix Verification ---
if __name__ == "__main__":
    print("\n================== LAUNCHING LANGGRAPH RUNTIME TEST ==================\n")
    
    SAMPLE_IMAGE = "E:\\rag-eval\\cystic acne.jpg"
    SAMPLE_TEXT = "Is it fine to clean my face with warm water daily?"
    test_config = {"configurable": {"thread_id": "test_user:thread_1"}}

    print("--- Testing Text Only with Supabase Persistence ---")
    case_2_output = skin_care_graph.invoke(
        {
            "messages": [("user", SAMPLE_TEXT)],
            "image_path": None,
            "user_query": SAMPLE_TEXT,
        },
        config=test_config,
    )
    print(f"Condition: {case_2_output['detected_condition']}\nAnswer: {case_2_output['final_response'][:150]}...\n")
