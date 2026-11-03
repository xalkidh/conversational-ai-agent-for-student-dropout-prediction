import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated
import operator

from src.tools import predict_dropout, retrieve_information, dataset_statistics

load_dotenv()

# Ορισμός state του agent
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# Αρχικοποίηση LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Σύνδεση tools με LLM
tools = [predict_dropout, retrieve_information, dataset_statistics]
llm_with_tools = llm.bind_tools(tools)

# Node 1: ο agent αποφασίζει τι να κάνει
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Node 2: εκτελεί το tool που επέλεξε ο agent
tool_node = ToolNode(tools)

# Συνάρτηση που αποφασίζει αν συνεχίζουμε ή σταματάμε
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Δημιουργία graph
def create_agent():
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END}
    )

    graph.add_edge("tools", "agent")

    return graph.compile()

agent = create_agent()

SYSTEM_PROMPT = (
    "You are an assistant for student dropout analysis. "
    "Always reply in the same language as the user's latest message (English or Greek). "
    "Use the available tools when needed."
)

def run_agent(message: str, history: list = []):
    # Add the system prompt only at the start of a session
    if not history:
        history = [SystemMessage(content=SYSTEM_PROMPT)]
    messages = history + [HumanMessage(content=message)]
    result = agent.invoke({"messages": messages})

    last_message = result["messages"][-1]
    content = last_message.content
    if isinstance(content, list):
        content = " ".join([c.get("text", "") for c in content if isinstance(c, dict)])
    return content, result["messages"]