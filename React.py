from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int):
    """This is an addition function that add 2 numbers together"""
    return a+b
@tool 
def subtract(a: int, b:int):
    """This is an subtraction function that subtract 2 numbers"""
    return a -  b
@tool 
def multiply(a: int, b:int):
    """This is an multiply function that subtract 2 numbers"""
    return a *  b


tools = [add, subtract, multiply]

model = ChatAnthropic(
    model_name='claude-opus-4-7',
    base_url='https://api.agentshop247.com/api/claude',
    temperature=0,
    timeout=None,
    stop=None,
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI asssistant, please answer my query to the best of your ability.")

    response = model.invoke([system_prompt] + state['messages'])
    return {'messages':[response]} 

def should_continue(state: AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else: return "continue"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'agent')
graph.add_conditional_edges(
    'agent', 
    should_continue,
    {
        'continue': 'tools',
        'end': END
    } 
) 

graph.add_edge('tools', 'agent')

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s['messages'][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {'messages': [('user', 'Add 1 + 2. Multiply that results by 6')]}
print_stream(app.stream(inputs, stream_mode = 'values'))
