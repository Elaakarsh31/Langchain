import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define directory containig the file
current_directory = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_directory, "books")
db_dir = os.path.join(current_directory, "db")
persistent_dir = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_dir}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The file {books_dir} does not exist. Please check the path."
        )
    
    # List all the files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    # Read the text content from the file
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

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
        docs, embeddings, persist_directory=persistent_dir
    )
    print("\n---Finished Creating vector store ---")

else:
    print("Vector store already exists. No need to initialize")