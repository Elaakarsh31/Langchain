import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define directory 
current_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_directory, "db", "chroma_db")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store 
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# User questoin:
query = input("Ask Question: ")

# retrieve relevant document 
retriever = db.as_retriever(search_type="similarity_score_threshold", 
                            search_kwargs={"k": 3, "score_threshold": 0.4})
relevant_docs = retriever.invoke(query)

# Display results
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unkown')}\n")