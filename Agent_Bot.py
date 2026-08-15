from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatAnthropic(
    model_name='claude-opus-4-7',
    base_url='https://api.agentshop247.com/api/claude',
    temperature=0,
    timeout=None,
    stop=None,
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            print(f"\nAI: {block.get('text', '')}")
            break
    return state

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START, "process",)
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != 'exit':
    agent.invoke({'messages': [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")

