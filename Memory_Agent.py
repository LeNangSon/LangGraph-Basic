import os
from typing import TypedDict, List
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages : List[BaseMessage]

llm = ChatAnthropic(
    model_name='claude-opus-4-7',
    base_url='https://api.agentshop247.com/api/claude',
    temperature=0,
    timeout=None,
    stop=None,
)


def process(state: AgentState) -> AgentState:
    full_response = llm.invoke(state['messages'])
    for block in full_response.content:
        if isinstance(block, dict) and block['type'] == 'text':
            response = block['text']
            break
    print(response)
    return {
        'messages': state['messages'] + [full_response]
    }

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START, "process",)
graph.add_edge("process", END)
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content = user_input))

    result = agent.invoke({'messages': conversation_history})

    conversation_history = result['messages']

    user_input = input("Enter: ")

file_name = 'logging.txt'
with open(file_name, 'w') as file:
    file.write("Your Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            for block in message.content:
                    if isinstance(block, dict) and block['type'] == 'text':
                        text = block['text']
                        break
            file.write(f"AI: {text}\n")
    file.write("End of Converstation")

print(f"Conversation saved to {file_name}")