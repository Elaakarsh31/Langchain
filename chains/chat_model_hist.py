from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# load the env
load_dotenv()

# Instantiate the model
model = ChatOpenAI(model="gpt-3.5-turbo")

chat_history = [] # use a list to store the chat history

# set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assitant.")
chat_history.append(system_message)

# chat loop
while True:
    query = input("You: ")
    if query.lower() =='exit':
        break
    chat_history.append(HumanMessage(content=query))

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print("AI: ",response)