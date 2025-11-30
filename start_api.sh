#!/bin/bash

# ============================================================
# FramePack Studio API Server Startup Script
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
# This prevents re-downloading them if you restart the pod.
mkdir -p /workspace/hf_cache
export HF_HOME=/workspace/hf_cache
echo "✅ HuggingFace cache set to /workspace/hf_cache"

# 5. API Configuration
# Set your API secret here or use environment variable
if [ -z "$FRAMEPACK_API_SECRET" ]; then
    export FRAMEPACK_API_SECRET="hasanhere"
    echo "⚠️  Using default API secret "hasanhere" - change FRAMEPACK_API_SECRET for production!"
else
    echo "✅ API secret loaded from environment"
fi

# Rate limiting (optional)
export RATE_LIMIT_REQUESTS=${RATE_LIMIT_REQUESTS:-20}
export RATE_LIMIT_WINDOW=${RATE_LIMIT_WINDOW:-60}
echo "✅ Rate limit: $RATE_LIMIT_REQUESTS requests per $RATE_LIMIT_WINDOW seconds"

# GPU Memory Preservation (20GB for RTX 4090)
export GPU_MEMORY_PRESERVATION=${GPU_MEMORY_PRESERVATION:-20}
echo "✅ GPU Memory Preservation: $GPU_MEMORY_PRESERVATION GB"

# 6. BASE URL for callbacks (IMPORTANT for webhooks!)
# Set this to your public RunPod URL
if [ -z "$BASE_URL" ]; then
    echo "⚠️  BASE_URL not set! Callbacks will use localhost (won't work externally)"
    echo "   Set it like: export BASE_URL=https://your-runpod-id-8000.proxy.runpod.net"
else
    export BASE_URL="$BASE_URL"
    echo "✅ BASE_URL set to: $BASE_URL"
fi

# 7. Launch the API
echo ""
echo "============================================================"
echo "🚀 Starting FramePack Studio API Server..."
echo "============================================================"
echo ""
echo "📡 API Endpoints:"
echo "   • Health Check: http://0.0.0.0:8000/health"
echo "   • API Docs:     http://0.0.0.0:8000/docs"
echo "   • Generate:     POST http://0.0.0.0:8000/generate"
echo ""
echo "🔑 Remember to set X-API-Key header in requests!"
echo ""
echo "(First run will download ~30GB models)"
echo "============================================================"
echo ""

python api.py
