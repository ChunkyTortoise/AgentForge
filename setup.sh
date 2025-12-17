#!/bin/bash

# AgentForge Setup Script
# Installs dependencies and verifies environment

echo "🔮 AgentForge Setup"
echo "==================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# 2. Virtual Environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment exists."
fi

# 3. Activate and Install
echo "⬇️  Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify
echo "🔍 Verifying installation..."
python3 tests/validate_modules.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup Complete! Run the app with:"
    echo "   source .venv/bin/activate"
    echo "   streamlit run app.py"
else
    echo ""
    echo "❌ Setup failed during verification."
fi
