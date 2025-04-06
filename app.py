# app.py
import streamlit as st
import sys
import os

# Add the current directory to the path so Python can find your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing agent files
from supervisor_agent import execute_workflow

# Page configuration
st.set_page_config(
    page_title="Retail Inventory Management System",
    page_icon="🏪",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main app function
def main():
    st.title("🏪 Retail Inventory Management System")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This AI-powered system helps retail managers make data-driven decisions by providing insights on:
        - 📊 Demand Forecasting
        - 📦 Inventory Monitoring
        - 💰 Price Optimization
        """)
        
        st.subheader("How it works")
        st.write("""
        1. Type your question in the chat
        2. A supervisor agent analyzes your query
        3. Specialized agents provide domain-specific responses
        """)
        
        # Example questions
        st.subheader("Example questions")
        example_questions = [
            "What's the forecast demand for product id 5321 next month?",
            "Which products should be understocker to avoid high charges for inventory",
            "How should we adjust pricing for seasonal items?",
            "What's our optimal inventory strategy for high-demand products?"
        ]
        
        for q in example_questions:
            if st.button(q):
                st.session_state.messages.append({"role": "user", "content": q})
                ask_question(q)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about demand, inventory, or pricing"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        ask_question(prompt)

def ask_question(question):
    """Process the user question and display the response"""
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            raw_response = execute_workflow(question)
            formatted_response = format_response(raw_response, question)
            st.markdown(formatted_response)

            # ✅ Append assistant's response to session state so it's remembered
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_response
            })


def format_response(raw_response, question):
    """Format the raw response from agents for better display"""
    # Split the response into parts from different agents
    parts = raw_response.split("\n\n")
    formatted_parts = []
    
    # Add the classification info if present (from supervisor)
    if "[🕵️‍♂️]" in raw_response:
        for part in parts:
            if "[🕵️‍♂️]" in part:
                category = part.strip()
                formatted_parts.append(f"**{category}**\n")
                break
    
    # Format each agent's response
    if "📊 Demand Forecasting Response:" in raw_response:
        content = raw_response.split("📊 Demand Forecasting Response:")[1]
        if "\n\n📦" in content:
            content = content.split("\n\n📦")[0]
        elif "\n\n💰" in content:
            content = content.split("\n\n💰")[0]
        formatted_parts.append(f"### 📊 Demand Forecasting\n{content.strip()}")
    
    if "📦 Inventory Monitoring Response:" in raw_response:
        content = raw_response.split("📦 Inventory Monitoring Response:")[1]
        if "\n\n📊" in content:
            content = content.split("\n\n📊")[0]
        elif "\n\n💰" in content:
            content = content.split("\n\n💰")[0]
        formatted_parts.append(f"### 📦 Inventory Monitoring\n{content.strip()}")
    
    if "💰 Pricing Optimization Response:" in raw_response:
        content = raw_response.split("💰 Pricing Optimization Response:")[1]
        if "\n\n📊" in content:
            content = content.split("\n\n📊")[0]
        elif "\n\n📦" in content:
            content = content.split("\n\n📦")[0]
        formatted_parts.append(f"### 💰 Price Optimization\n{content.strip()}")
    
    # If no parts were formatted, return the raw response
    if not formatted_parts:
        return raw_response
    
    # Join all parts with line breaks
    return "\n\n".join(formatted_parts)

if __name__ == "__main__":
    main()