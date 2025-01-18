import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from services.chat_agent import ChatManager

def initialize_session_state():
    """Initialize session state variables."""
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Socratic RAG Agent")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What research topics are you interested in?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            response = st.session_state.chat_manager.handle_message(prompt)
            st.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add a reset button in the sidebar
    with st.sidebar:
        if st.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.chat_manager.reset_conversation()
            st.rerun()

if __name__ == "__main__":
    main() 