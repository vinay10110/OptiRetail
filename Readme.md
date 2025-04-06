# Retail Inventory Management System using LangGraph Swarm

This project is a multi-agent AI system for retail inventory management, built using LangGraph Swarm architecture. The system helps predict demand, ensure product availability, reduce inventory holding costs, and improve supply chain efficiency.

## Architecture Overview

The system uses a swarm of specialized agents that collaborate to provide comprehensive insights:

1. **Supervisor Agent**: Analyzes user queries and determines which specialized agents to activate
2. **Demand Forecasting Agent**: Predicts future product demand based on historical data
3. **Inventory Monitoring Agent**: Tracks inventory levels and identifies potential stockout issues
4. **Price Optimization Agent**: Recommends pricing strategies based on demand and competition

These agents communicate through dedicated channels and coordinate to provide a unified response.

## Key Features

- **Dynamic Agent Selection**: The supervisor activates only the relevant agents based on query analysis
- **Inter-Agent Communication**: Agents can share information and collaborate on complex problems
- **Parallel Processing**: Multiple agents can work simultaneously on different aspects of a query
- **Coordinated Response**: Results from all agents are synthesized into a comprehensive final report
- **Streamlit UI**: User-friendly interface for interacting with the system

## Technical Implementation

The system is built using:

- **LangGraph Swarm**: For multi-agent orchestration and communication
- **LangChain**: For document processing, embeddings, and retrieval
- **FAISS**: For efficient vector storage and similarity search
- **Ollama**: For local LLM inference
- **Streamlit**: For the web interface

## Getting Started

### Prerequisites

- Python 3.8+
- Ollama with the following models configured:
  - `nomic-embed-text` (for embeddings)
  - `tinyllama` (for specialized agents)
  - `gemma3` (for the supervisor)

### Installation

1. Clone this repository
2. Install the requirements:
   ```
   pip install langgraph-swarm langchain langchain-ollama faiss-cpu streamlit
   ```
3. Prepare your CSV datasets in the specified locations or update the paths in the configuration section

### Running the Application

To start the Streamlit app:

```
streamlit run app.py
```

## System Workflow

1. User submits a query through the Streamlit interface
2. The supervisor agent analyzes the query and determines which specialized agents to activate
3. The activated agents process the query in parallel, accessing their respective knowledge bases
4. Each agent generates a report with its findings and confidence score
5. The final report generator compiles all findings into a comprehensive response
6. The response is displayed to the user in a formatted manner

## Directory Structure

```
retail-inventory-system/
├── app.py                           # Streamlit interface
├── langgraph_swarm_retail.py        # Main LangGraph Swarm implementation
├── datasets/                        # CSV data files
│   ├── demand_forecasting.csv
│   ├── inventory_monitoring.csv
│   └── pricing_optimization.csv
└── faiss_indices/                   # FAISS vector indices (generated on first run)
    ├── demand_faiss_index/
    ├── inventory_faiss_index/
    └── pricing_faiss_index/
```

## Extending the System

To add new capabilities:

1. Create a new specialized agent in the langgraph_swarm_retail.py file
2. Add the agent to the supervisor's list of available agents
3. Update the swarm graph to include the new agent in the workflow
4. Add any new data sources and create the corresponding FAISS indices

## Benefits Over Previous Implementation

Compared to the original architecture, this LangGraph Swarm implementation offers:

- More dynamic agent selection based on query context
- Better inter-agent communication capabilities
- Improved parallelization of agent tasks
- More structured state management through the StateGraph
- Enhanced scalability for adding new specialized agents
- More intuitive visualization of the processing workflow in the UI