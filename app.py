from langchain_ollama import OllamaEmbeddings, OllamaLLM
import chromadb
import os
import streamlit as st

# Configuration
llm_model = "llama3.2"
base_url = "http://localhost:11434"  # Adjust the base URL for your Ollama server

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path=os.path.join(os.getcwd(), "chroma_db"))

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    def __init__(self, langchain_embeddings):
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.langchain_embeddings.embed_documents(input)

# Initialize the embedding function
embedding = ChromaDBEmbeddingFunction(
    OllamaEmbeddings(model=llm_model, base_url=base_url)
)

# Define or get the ChromaDB collection
collection_name = "rag_collection_demo_1"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"description": "A collection for RAG with Ollama - Demo1"},
    embedding_function=embedding
)

# Function to add documents to the ChromaDB collection
def add_documents_to_collection(documents, ids):
    collection.add(documents=documents, ids=ids)

# Add sample documents (only run this when initializing the collection)
if "initialized" not in st.session_state:
    documents = [
        "Artificial intelligence is the simulation of human intelligence processes by machines.",
        "Python is a programming language that lets you work quickly and integrate systems more effectively.",
        "ChromaDB is a vector database designed for AI applications."
    ]
    doc_ids = ["doc1", "doc2", "doc3"]
    add_documents_to_collection(documents, doc_ids)
    st.session_state["initialized"] = True

# Function to query ChromaDB
def query_chromadb(query_text, n_results=1):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    return results["documents"], results["metadatas"]

# Function to interact with the Ollama LLM
def query_ollama(prompt):
    llm = OllamaLLM(model=llm_model, base_url=base_url)
    return llm.stream(prompt)

# RAG pipeline: Combine ChromaDB and Ollama
def rag_pipeline(query_text):
    # Retrieve relevant documents
    retrieved_docs, metadata = query_chromadb(query_text)
    context = " ".join(retrieved_docs[0]) if retrieved_docs else "No relevant documents found."

    # Augment the prompt with context
    augmented_prompt = f"Context: {context}\n\nQuestion: {query_text}\nAnswer:"
    response = query_ollama(augmented_prompt)
    return response

# Streamlit UI
st.title("Chatbot using Ollama llama3.2 with RAG")
prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            response = rag_pipeline(prompt)
            st.write("### Response:")
            st.write(response)
#Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#.\lama\Scripts\Activate.ps1