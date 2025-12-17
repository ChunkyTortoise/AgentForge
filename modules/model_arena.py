"""
Model Arena Module - LLM Comparison.

Demonstrates:
- Duke LLMOps Specialization
- DeepLearning.AI Deep Learning
- Multi-model evaluation
"""
import time

import streamlit as st
from core.llm_client import LLMClient, get_available_providers

from utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """Render the Model Arena module."""
    st.markdown("""
    Compare LLM responses side-by-side. Evaluate quality, speed,
    and cost across different models.

    **Skills Demonstrated**: Duke LLMOps, DeepLearning.AI
    """)

    # Check providers
    providers = get_available_providers()

    st.markdown("### 🔌 Available Providers")
    col1, col2 = st.columns(2)
    with col1:
        status = "✅ Ready" if providers.get("gemini") else "❌ Not configured"
        st.markdown(f"**Google Gemini**: {status}")
    with col2:
        status = "✅ Ready" if providers.get("claude") else "❌ Not configured"
        st.markdown(f"**Anthropic Claude**: {status}")

    if not any(providers.values()):
        st.warning("Configure at least one provider in your `.env` file")
        return

    st.markdown("---")
    st.markdown("### ⚔️ Model Comparison")

    prompt = st.text_area(
        "Enter a prompt to compare",
        height=100,
        placeholder="Explain quantum computing in simple terms..."
    )

    if st.button("🚀 Compare Models", disabled=not prompt):
        available_providers = [k for k, v in providers.items() if v]

        cols = st.columns(len(available_providers))

        for i, provider in enumerate(available_providers):
            with cols[i]:
                st.markdown(f"### {provider.title()}")
                try:
                    client = LLMClient(provider=provider)
                    start = time.time()
                    response = client.generate(prompt)
                    elapsed = time.time() - start

                    st.markdown(response.content)
                    st.caption(f"⏱️ {elapsed:.2f}s | Model: {response.model}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📊 Model Comparison Matrix")
    st.markdown("""
    | Feature | Gemini 1.5 Flash | Claude 3.5 Sonnet |
    |---------|-----------------|-------------------|
    | Speed | ⚡ Very Fast | 🚀 Fast |
    | Cost | 💚 Free tier | 💛 $3/$15 per MTok |
    | Context | 1M tokens | 200K tokens |
    | Best For | General tasks | Complex reasoning |
    """)
