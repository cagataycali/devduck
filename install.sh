#!/bin/bash
# 🦆 DevDuck installer - Extreme minimalist agent

echo "🦆 Installing Devduck..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check Ollama  
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        # Linux
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
fi

# Start ollama service
echo "🦆 Starting Ollama service..."
ollama serve &
sleep 2

# Pull a basic model
echo "🦆 Pulling basic model..."
ollama pull qwen3:1.7b

# Test devduck
echo "🦆 Testing Devduck..."
python3 devduck/__init__.py "what's 5*7?"

echo "✅ Devduck installed successfully!"
echo "Usage: python3 devduck/__init__.py 'your question'"