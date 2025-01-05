import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define directory containig the file
current_directory = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_directory, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# Define the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Function to query a vector store
def query_vector_store(store_name, query, embedding_function, search_type, search_kwargs):

    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_function)
        retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        relevant_docs = retriever.invoke(query)
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs,1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector Store {store_name} does not exist.")

# Define query
query = "How did Juliet Die?"

# Results with Similarity Search
print("\n--- Using Similarity Search ---")
query_vector_store("chroma_db_with_metadata", query, embeddings, "similarity", {"k": 3})

# Results with Max Marginal Relevance
print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store("chroma_db_with_metadata", 
                   query, embeddings, 
                   "mmr", 
                   {"k": 3, "fetch_k": 20, "lambda_mult": 0.5})    

# Results with Similarity Score
print("\n--- Using Similarity Score Threshold ---")
query_vector_store("chroma_db_with_metadata", 
                   query, embeddings, 
                   "similarity_score_threshold", 
                   {"k": 3, "score_threshold": 0.1})  