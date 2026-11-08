from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Defines the state structure passed between LangGraph nodes."""
    messages: Annotated[List[AnyMessage], add_messages]
    user_query: Optional[str]
    image_path: Optional[str]
    detected_condition: Optional[str]
    retrieved_context: Optional[List[str]]
    final_response: Optional[str]
