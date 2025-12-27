"""
Streaming Chat Module - Real-time LLM interaction.

Demonstrates:
- Server-Sent Events (SSE) handling in Streamlit
- Async streaming from FastAPI backend
- Responsive UI for long-running LLM tasks
"""
import streamlit as st
import requests
import os
import json
from utils.logger import get_logger

logger = get_logger(__name__)

def render():
    st.markdown("## 💬 Streaming Chat")
    st.markdown("Experience real-time token streaming from the AgentForge backend.")

    # Sidebar settings for this module
    with st.sidebar.expander("Chat Settings", expanded=True):
        provider = st.selectbox("Model Provider", ["gemini", "claude"], key="chat_provider")
        system_prompt = st.text_area("System Prompt", 
                                   "You are a helpful AI assistant in the AgentForge platform. Respond concisely.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat container
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Call the streaming endpoint
            try:
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                payload = {
                    "message": prompt,
                    "provider": provider,
                    "system_prompt": system_prompt
                }
                
                # Use requests with stream=True
                with requests.post(
                    f"{backend_url}/chat/stream", 
                    json=payload, 
                    stream=True, 
                    timeout=60
                ) as r:
                    if r.status_code == 200:
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                full_response += chunk
                                response_placeholder.markdown(full_response + "▌")
                        
                        # Final update without cursor
                        response_placeholder.markdown(full_response)
                        # Save to history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        st.error(f"Error from backend: {r.status_code}")
                        st.write(r.text)
            
            except Exception as e:
                st.error(f"Streaming failed: {e}")
                logger.error(f"Chat streaming error: {e}", exc_info=True)

    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
