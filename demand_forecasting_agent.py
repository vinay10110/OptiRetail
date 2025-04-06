# ---------------------
# IMPORTS
# ---------------------
import pandas as pd
import os
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader

# ---------------------
# CONFIGURATION
# ---------------------
CSV_FILE = r'Z:\projects\optimizing retail inventory\datasets\demand_forecasting.csv'
FAISS_INDEX_FOLDER = r'Z:\projects\optimizing retail inventory\demand_faiss_index'  # Folder to store FAISS index

# ---------------------
# STEP 1: CREATE DEMAND FORECASTING AGENT
# ---------------------
def create_demand_agent(csv_path):
    embedding = OllamaEmbeddings(model="nomic-embed-text")

    # Try to load existing embeddings first
    try:
        print(f"[⏳] Attempting to load existing FAISS index from {FAISS_INDEX_FOLDER}...")
        vs = FAISS.load_local(FAISS_INDEX_FOLDER, embedding)
        print(f"[✓] Successfully loaded {len(vs.index_to_docstore_id)} vectors from FAISS index")
    except (FileNotFoundError, ValueError, Exception) as e:
        # If loading fails, create new embeddings
        print(f"[!] Could not load existing FAISS index: {e}")
        print("[⏳] Creating new embeddings...")
        vs = create_new_embeddings(csv_path, embedding)

    retriever = vs.as_retriever()
    llm = OllamaLLM(model="tinyllama")  # Change model if needed

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def create_new_embeddings(csv_path, embedding):
    # Create directory if it doesn't exist
    os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)
    
    # Load & Split the CSV into documents
    print(f"[⏳] Loading documents from {csv_path}...")
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    print(f"[✓] Loaded {len(docs)} documents.")
    
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    print("[⏳] Splitting documents...")
    split_docs = splitter.split_documents(docs)
    print(f"[✓] Created {len(split_docs)} document chunks.")

    # Create new embeddings with FAISS
    print("[⏳] Creating new embeddings with FAISS (this may take some time)...")
    vs = FAISS.from_documents(split_docs, embedding)
    
    # Save the FAISS index
    print(f"[⏳] Saving FAISS index to {FAISS_INDEX_FOLDER}...")
    vs.save_local(FAISS_INDEX_FOLDER)

    print(f"[✓] Created and stored {len(split_docs)} embeddings in FAISS index.")
    return vs

# ---------------------
# MAIN EXECUTION
# ---------------------
print("=" * 50)
print("DEMAND FORECASTING AGENT")
print("=" * 50)

# Instantiate the demand agent (fast startup if embeddings exist)
try:
    print("[⏳] Initializing demand forecasting agent...")
    demand_agent = create_demand_agent(CSV_FILE)
    print("[✓] Agent ready!")
except Exception as e:
    print(f"[ERROR] Failed to initialize agent: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# ---------------------
# STEP 2: QUERY THE DEMAND FORECASTING AGENT
# ---------------------
def query_demand_forecasting(question):
    print(f"\n[⏳] Processing query: \"{question}\"")
    result = demand_agent.invoke({"query": question})
    
    # Create a formatted response string that includes the agent's header
    response = "📊 Demand Forecasting Response:\n\n"
    response += result["result"]
    
    # Print to console (for debugging and standalone use)
    print("\n" + response)
    print("\n" + "=" * 50 + "\n")
    
    # Return the response (to be captured by the supervisor)
    return response

# Interactive loop to ask questions (only runs when file is executed directly)
if __name__ == "__main__":
    print("\nDemand Forecasting Agent is ready! Type 'exit' to quit.")
    while True:
        user_question = input("\nAsk a question about demand forecasting: ")
        if user_question.lower() in ["exit", "quit", "q"]:
            print("Exiting. Goodbye!")
            break
        query_demand_forecasting(user_question)