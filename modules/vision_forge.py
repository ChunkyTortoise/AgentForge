"""
Vision Forge Module - Multimodal AI.

Capabilities:
1. Chat with Images (Iterative)
2. Video Analysis (Gemini Native)
3. Structured Data Extraction (JSON)
"""
import streamlit as st
import base64
import json
import os
from PIL import Image

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from core.llm_client import get_available_providers
from utils.logger import get_logger

logger = get_logger(__name__)

def render() -> None:
    st.markdown("## 👁️ Vision Forge: Multimodal Intelligence")
    
    # Session State Init
    if "vision_chat_history" not in st.session_state:
        st.session_state.vision_chat_history = []
    if "vision_current_image" not in st.session_state:
        st.session_state.vision_current_image = None
    
    # Tabs
    tab_chat, tab_video, tab_extract = st.tabs(["💬 Chat with Image", "📹 Video Intelligence", "🧾 Data Extraction"])
    
    # --- TAB 1: CHAT WITH IMAGE ---
    with tab_chat:
        st.markdown("### Interactive Visual Analysis")
        
        # 1. Config
        col_c1, col_c2 = st.columns([1, 3])
        with col_c1:
            provider = st.selectbox("Model", ["gemini", "claude"], key="v_chat_provider")
            
        with col_c2:
            uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], key="v_chat_upload")

        # Handle Upload
        if uploaded_img:
            # Check if new image
            file_key = f"img_{uploaded_img.name}"
            if st.session_state.vision_current_image != file_key:
                st.session_state.vision_current_image = file_key
                st.session_state.vision_chat_history = [] # Reset chat on new image
                # Convert to base64 once
                st.session_state.vision_b64 = _file_to_base64(uploaded_img)
                st.session_state.vision_mime = uploaded_img.type
            
            # Display
            st.image(uploaded_img, caption="Current Image", width=400)
            
            # Chat Loop
            for msg in st.session_state.vision_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Input
            if prompt := st.chat_input("Ask about this image..."):
                # User Msg
                st.session_state.vision_chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Agent Msg
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        response = _run_vision_chat(
                            provider, 
                            prompt, 
                            st.session_state.vision_b64, 
                            st.session_state.vision_mime,
                            st.session_state.vision_chat_history[:-1] # History excluding current
                        )
                        st.markdown(response)
                        st.session_state.vision_chat_history.append({"role": "assistant", "content": response})
        else:
            st.info("Upload an image to start chatting.")

    # --- TAB 2: VIDEO INTELLIGENCE ---
    with tab_video:
        st.markdown("### 📹 Video Analysis (Gemini)")
        st.warning("Requires Google Gemini 1.5 Pro/Flash.")
        
        video_file = st.file_uploader("Upload Video", type=["mp4", "mov"], key="v_video_upload")
        
        if video_file:
            st.video(video_file)
            
            v_prompt = st.text_input("Question", value="Describe the events in this video in detail.")
            
            if st.button("Analyze Video"):
                with st.spinner("Gemini is watching your video... (This Make Take 10-20s)"):
                    try:
                        # Direct Gemini Call for Video
                        import google.generativeai as genai
                        
                        # Save temp
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                            tmp.write(video_file.getvalue())
                            tmp_path = tmp.name
                        
                        # Upload to Gemini File API
                        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                        video_asset = genai.upload_file(tmp_path)
                        
                        # Wait for processing
                        import time
                        while video_asset.state.name == "PROCESSING":
                            time.sleep(2)
                            video_asset = genai.get_file(video_asset.name)
                            
                        if video_asset.state.name == "FAILED":
                            st.error("Video processing failed.")
                        else:
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            res = model.generate_content([video_asset, v_prompt])
                            st.markdown(res.text)
                            
                        # Cleanup
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"Video Error: {e}")

    # --- TAB 3: EXTRACTION ---
    with tab_extract:
        st.markdown("### 🧾 Structured Data Extraction")
        st.markdown("Extract precise JSON from documents (Invoices, Receipts, Forms).")
        
        ext_file = st.file_uploader("Upload Document", type=["jpg", "png", "pdf"], key="v_ext_upload")
        
        if ext_file:
            st.image(ext_file, width=300)
            fields = st.text_input("Fields to extract (comma separated)", value="Invoice_Number, Date, Total_Amount, Vendor_Name")
            
            if st.button("Extract Data"):
                with st.spinner("Extracting..."):
                    try:
                        # 1. Base64
                        b64 = _file_to_base64(ext_file)
                        mime = ext_file.type
                        
                        # 2. Prompt for JSON
                        prompt = f"""
                        Extract the following fields: {fields}.
                        Return ONLY a valid JSON object. No markdown.
                        If a field is missing, use null.
                        """
                        
                        # 3. Call (Prefer Gemini for JSON mode usually, or Claude)
                        res = _run_vision_chat("gemini", prompt, b64, mime, [])
                        
                        # 4. Parse & Show
                        # Clean cleanup
                        cleaned = res.replace("```json", "").replace("```", "").strip()
                        data = json.loads(cleaned)
                        
                        st.json(data)
                        
                    except Exception as e:
                        st.error(f"Extraction Error: {e}")


def _file_to_base64(file):
    return base64.b64encode(file.getvalue()).decode('utf-8')

def _run_vision_chat(provider, prompt, b64, mime, history):
    """Unified Vision Chat Runner."""
    
    # Construct Messages
    # Note: History in LangChain multimodal is tricky. 
    # For now, we will just send [Image, Prompt] as a fresh call 
    # but append previous conversation as context TEXT if needed.
    # To keep it simple and robust: Single turn with history as text context?
    # Or Multi-turn if supported.
    
    # Gemini supports multi-turn with images.
    
    messages = []
    
    # 1. Add Image to the FIRST message (System/Human)
    content_block = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    ]
    
    # If we have history, we might want to just append it as text context 
    # because re-sending base64 every time is expensive/redundant.
    if history:
        hist_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])
        content_block[0]["text"] = f"Context:\n{hist_text}\n\nCurrent Question: {prompt}"

    final_msg = HumanMessage(content=content_block)
    
    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
        res = llm.invoke([final_msg])
        return res.content
        
    elif provider == "claude":
        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0)
        res = llm.invoke([final_msg])
        return res.content
        
    return "Error: Provider unavailable."
