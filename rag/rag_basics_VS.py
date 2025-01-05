import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define directory containig the file
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, "books", "odyssey.txt")
persistent_directory = os.path.join(current_directory, "db", "chroma_db")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    # Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about split documents
    print("\n--- Document Chunks Information ---")
    print(f"Length of Dcoument: {len(documents[0].page_content)}")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create Embeddings
    print("\n--- Creating Embeddings ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("\n--- Finished Creating Embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n---Finished Creating vector store ---")

else:
    print("Vector store already exists. No need to initialize")