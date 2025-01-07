import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# load environment variables
load_dotenv()

# Define Vector database directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

# Define embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# load existing vector embeddings
db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Create retreiver for querying vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create chat model
llm = ChatOpenAI(model="gpt-4o")

# Contextualize question prompts
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the history, "
    "formulate a standalone question which can be understood "
    "without the chat history. DO NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create prompt template
contextualize_q_prompt = ChatPromptTemplate([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create history aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Setup QnA prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use the following pieces of "
    "retreived context to answer the question. If you don't know the answer. just say that "
    "you don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

# Create prompt for QnA
qa_prompt = ChatPromptTemplate([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create chain to feed all documents context to llm
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create retrieval chain that combines history-aware and qna chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Conversional
def continual_chat():
    print("Starting Chat with AI! Type 'exit' to end.")
    chat_history = []
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        # Display AI response
        print(f"AI: {result['answer']}")

        # Update chat history
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result['answer']))

if __name__ == "__main__":
    continual_chat()