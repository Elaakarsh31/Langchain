from dotenv import load_dotenv
from google.cloud import firestore
from langchain_openai import ChatOpenAI
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# load the env
load_dotenv()

# Setup Firebase Firestore
PROJECT_ID = "chat-4ff6e"
SESSION_ID = "user_session"
COLLECTION_NAME = "chat_history"

# Initialize Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project= PROJECT_ID)

# Initializr chat history
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client = client
)
print("Chat History Initialized.")
print("Current Chat History: ", chat_history.messages)

# model instantiate
model = ChatOpenAI()

print("Start chatting with AI. Type 'exit' to quit.")

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    chat_history.add_user_message(query)

    response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(response.content)

    print(f"AI: {response.content}")

# print("------ Message History ------")
# print(chat_history)