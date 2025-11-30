#!/bin/bash

# ============================================================
# FramePack Studio - Run BOTH API and Gradio UI
# For RunPod / Linux environments
# ============================================================

# 1. Navigate to the correct folder
cd /workspace/FramePack-Studio

# 2. Activate the virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No venv found, using system Python"
fi

# 3. OPTIMIZATION: Prevent Out-Of-Memory crashes on 4090
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "✅ PyTorch memory optimization enabled"

# 4. STORAGE: Force all models to go to the persistent volume
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
echo "✅ HuggingFace cache set to /workspace/hf_cache"

# 5. API Configuration
if [ -z "$FRAMEPACK_API_SECRET" ]; then
    export FRAMEPACK_API_SECRET="your_secure_api_key_here"
    echo "⚠️  Using default API secret - change FRAMEPACK_API_SECRET for production!"
else
    echo "✅ API secret loaded from environment"
fi

export RATE_LIMIT_REQUESTS=${RATE_LIMIT_REQUESTS:-20}
export RATE_LIMIT_WINDOW=${RATE_LIMIT_WINDOW:-60}

# 6. Choose what to run
echo ""
echo "============================================================"
echo "    FramePack Studio Launcher"
echo "============================================================"
echo ""
echo "  1) API Server only     (port 8000)"
echo "  2) Gradio UI only      (port 7860)"
echo "  3) Both API + Gradio   (ports 8000 + 7860)"
echo ""
read -p "Select option [1-3]: " choice

case $choice in
    1)
        echo "🚀 Starting API Server..."
        python api.py
        ;;
    2)
        echo "🚀 Starting Gradio UI..."
        python studio.py --server 0.0.0.0 --port 7860 --share
        ;;
    3)
        echo "🚀 Starting Both Services..."
        echo ""
        echo "⚠️  NOTE: Running both requires ~48GB+ VRAM or will share models"
        echo "    API will be on port 8000, Gradio on port 7860"
        echo ""
        
        # Start API in background
        python api.py &
        API_PID=$!
        echo "✅ API Server started (PID: $API_PID)"
        
        # Wait a moment for API to initialize
        sleep 5
        
        # Start Gradio in foreground
        python studio.py --server 0.0.0.0 --port 7860 --share
        
        # If Gradio exits, also stop API
        kill $API_PID 2>/dev/null
        ;;
    *)
        echo "Invalid option. Starting API Server by default..."
        python api.py
        ;;
esac
