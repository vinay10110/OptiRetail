import pandas as pd
import os
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader

CSV_FILE = r'Z:\projects\optimizing retail inventory\datasets\inventory_monitoring.csv'
FAISS_INDEX_FOLDER = r'Z:\projects\optimizing retail inventory\inventory_faiss_index'

def create_inventory_agent(csv_path):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    try:
        print("[⏳] Loading existing FAISS index...")
        vs = FAISS.load_local(FAISS_INDEX_FOLDER, embedding)
        print(f"[✓] Successfully loaded {len(vs.index_to_docstore_id)} vectors from FAISS index")
    except:
        print("[!] FAISS index not found, creating new one...")
        vs = create_new_embeddings(csv_path, embedding)

    retriever = vs.as_retriever()
    llm = OllamaLLM(model="tinyllama")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def create_new_embeddings(csv_path, embedding):
    os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)
    print(f"[⏳] Loading documents from {csv_path}...")
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    print(f"[✓] Loaded {len(docs)} documents.")
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("[⏳] Splitting documents...")
    split_docs = splitter.split_documents(docs)
    print(f"[✓] Created {len(split_docs)} document chunks.")

    print("[⏳] Creating new embeddings with FAISS...")
    vs = FAISS.from_documents(split_docs, embedding)
    
    print(f"[⏳] Saving FAISS index to {FAISS_INDEX_FOLDER}...")
    vs.save_local(FAISS_INDEX_FOLDER)
    print(f"[✓] Created and stored {len(split_docs)} embeddings in FAISS index.")
    
    return vs

print("=" * 50)
print("INVENTORY MONITORING AGENT")
print("=" * 50)

# Instantiate the inventory agent
try:
    print("[⏳] Initializing inventory monitoring agent...")
    inventory_agent = create_inventory_agent(CSV_FILE)
    print("[✓] Agent ready!")
except Exception as e:
    print(f"[ERROR] Failed to initialize agent: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

def query_inventory(question):
    print(f"\n[⏳] Processing inventory query: \"{question}\"")
    result = inventory_agent.invoke({"query": question})
    
    # Create a formatted response string that includes the agent's header
    response = "📦 Inventory Monitoring Response:\n\n"
    response += result["result"]
    
    # Print to console (for debugging and standalone use)
    print("\n" + response)
    print("\n" + "=" * 50 + "\n")
    
    # Return the response (to be captured by the supervisor)
    return response

if __name__ == "__main__":
    print("\nInventory Monitoring Agent is ready! Type 'exit' to quit.")
    while True:
        user_question = input("\nAsk about inventory levels: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        query_inventory(user_question)