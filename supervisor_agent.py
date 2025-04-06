from langchain_ollama.llms import OllamaLLM
from demand_forecasting_agent import query_demand_forecasting
from inventory_monitoring_agent import query_inventory
from price_optimization_agent import query_pricing

# Initialize the supervisor LLM
supervisor_llm = OllamaLLM(model="gemma3")  # Use a strong model for reasoning

def classify_query(user_question):
    """Uses an LLM to classify the user's query into a relevant category."""
    prompt = f"""
    Classify the following user question into one of these categories:
    1. Demand Forecasting
    2. Inventory Monitoring
    3. Price Optimization
    4. Multiple Categories (if it involves more than one)

    User question: "{user_question}"
    Output the category name only.
    """
    category = supervisor_llm.invoke(prompt)
    return category.strip().lower()

def execute_workflow(user_question):
    """Determines which agent(s) to call based on the query classification."""
    category = classify_query(user_question)
    
    # Create a classification header
    result = f"[🕵️‍♂️] Supervisor classified query as: {category.capitalize()}\n\n"
    
    # Track which agents were called for better UI organization
    agents_called = []

    if "demand forecasting" in category or "multiple" in category:
        demand_response = query_demand_forecasting(user_question)
        result += demand_response + "\n\n"
        agents_called.append("demand")

    if "inventory monitoring" in category or "multiple" in category:
        inventory_response = query_inventory(user_question)
        result += inventory_response + "\n\n"
        agents_called.append("inventory")

    if "price optimization" in category or "multiple" in category:
        pricing_response = query_pricing(user_question)
        result += pricing_response + "\n\n"
        agents_called.append("pricing")

    # If no specific category was matched, default to using all agents
    if not agents_called:
        print("\n[⚠️] Category not specifically matched. Using all agents as fallback.\n")
        
        demand_response = query_demand_forecasting(user_question)
        result += demand_response + "\n\n"
        
        inventory_response = query_inventory(user_question)
        result += inventory_response + "\n\n"
        
        pricing_response = query_pricing(user_question)
        result += pricing_response + "\n\n"

    return result