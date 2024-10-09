from custom_chat_model import CustomChatModelAdvanced
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, AIMessage

model_name = "AI-Sweden-Models/gpt-sw3-1.3b-instruct"
model = CustomChatModelAdvanced(n=3, model_name=model_name)

result = model.invoke(
    [
        HumanMessage(content="hello!"),
        AIMessage(content="Hi there human!"),
        HumanMessage(content="Meow!"),
    ]
)

print(result)