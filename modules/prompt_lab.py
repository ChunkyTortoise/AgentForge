"""
Prompt Lab Module - Prompt Engineering Tools.

Demonstrates:
- Vanderbilt Prompt Engineering (18h)
- Dynamic Variable Injection
- Cost Estimation & A/B Testing
"""
import re
import streamlit as st
import time

from core.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the Prompt Lab module."""
    st.markdown("## ⚡ Prompt Lab: Engineering & Optimization")
    st.markdown("""
    Design, Test, and Optimize LLM prompts with **Dynamic Variables** and **Cost Analysis**.
    """)

    # 1. State Management
    if "prompt_templates" not in st.session_state:
        st.session_state.prompt_templates = {
            "Summarize": "Summarize the following text in {length} sentences:\n\n{text}",
            "Classification": "Classify the following email as 'Spam', 'Sales', or 'Support':\n\nSubject: {subject}\nBody: {body}",
            "Code Doc": "Write numpy-style docstrings for this python function:\n\n{code}"
        }

    # Tabs
    tab_editor, tab_arena = st.tabs(["✍️ Prompt Editor", "⚔️ Model Arena"])

    # --- EDITOR TAB ---
    with tab_editor:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("1. Design")
            
            # Template Loader
            template_names = list(st.session_state.prompt_templates.keys())
            selected_template = st.selectbox("Load Template", ["<New>"] + template_names)
            
            default_text = ""
            if selected_template != "<New>":
                default_text = st.session_state.prompt_templates[selected_template]

            prompt_text = st.text_area(
                "Prompt Template (Use {var} for placeholders)", 
                value=default_text,
                height=300,
                key="editor_prompt"
            )

            # Variable Detection
            variables = sorted(list(set(re.findall(r'\{(\w+)\}', prompt_text))))
            
            inputs = {}
            if variables:
                st.markdown("#### Variables")
                for var in variables:
                    inputs[var] = st.text_input(f"{var}", key=f"var_{var}")

             # Save Template
            with st.expander("Save Template"):
                new_name = st.text_input("Template Name")
                if st.button("Save to Library"):
                    if new_name and prompt_text:
                        st.session_state.prompt_templates[new_name] = prompt_text
                        st.success(f"Saved '{new_name}'!")
                        st.rerun()

        with col_right:
            st.subheader("2. Test & Evaluate")
            
            # Config
            col_p, col_m = st.columns(2)
            with col_p:
                provider = st.selectbox("Provider", ["gemini", "claude"], key="lab_provider")
            with col_m:
                model = st.text_input("Model ID (Opt)", placeholder="default")

            # Preview
            final_prompt = prompt_text
            try:
                final_prompt = prompt_text.format(**inputs)
            except KeyError:
                pass # Wait for inputs
            except Exception:
                pass 
                
            st.markdown("**Preview:**")
            st.code(final_prompt, language="text")
            
            # Cost Estimate (Rough)
            est_tokens = len(final_prompt) / 4
            cost = 0.0
            if provider == "claude":
                cost = (est_tokens / 1_000_000) * 3.00 # $3/1M input for Sonnet roughly
            st.caption(f"Est. Input Tokens: ~{int(est_tokens)} | Cost: ${cost:.6f}")

            if st.button("🚀 Run Experiment", type="primary"):
                if not final_prompt.strip():
                    st.warning("Prompt is empty.")
                else:
                    try:
                        client = LLMClient(provider=provider, model=model if model else None)
                        if not client.is_available():
                            st.error("Provider not configured.")
                        else:
                            with st.spinner("Generating..."):
                                start = time.time()
                                response = client.generate(final_prompt)
                                lat = time.time() - start
                                
                                st.success(f"Finished in {lat:.2f}s")
                                st.markdown("### Output")
                                st.markdown(response.content)
                                
                                st.markdown("---")
                                st.json({
                                    "model": response.model,
                                    "tokens_used": response.tokens_used,
                                    "latency_sec": round(lat, 3)
                                })
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- ARENA TAB ---
    with tab_arena:
        st.markdown("### ⚔️ Battle Mode")
        st.markdown("Compare two models side-by-side.")
        
        arena_prompt = st.text_area("Test Prompt", height=150, key="arena_prompt_input")
        
        if st.button("🥊 Fight!"):
            c1, c2 = st.columns(2)
            
            # GEMINI
            with c1:
                st.markdown("#### 💎 Gemini")
                try:
                    g_client = LLMClient("gemini")
                    if g_client.is_available():
                        with st.spinner("Gemini generating..."):
                            g_resp = g_client.generate(arena_prompt)
                            st.markdown(g_resp.content)
                            st.caption(f"Time: {0.0}s (Demo)") # Placeholder timing
                    else:
                        st.error("Gemini Missing")
                except Exception as e:
                    st.error(str(e))
                    
            # CLAUDE
            with c2:
                st.markdown("#### 🧠 Claude")
                try:
                    c_client = LLMClient("claude")
                    if c_client.is_available():
                        with st.spinner("Claude generating..."):
                            c_resp = c_client.generate(arena_prompt)
                            st.markdown(c_resp.content)
                    else:
                        st.error("Claude Missing")
                except Exception as e:
                    st.error(str(e))

