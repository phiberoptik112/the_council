#!/bin/bash
#
# The Council - Initialization Script
# Starts the Django-Q worker and development server
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "  The Council - Multi-Model LLM Voting"
echo "=========================================="
echo ""

# Activate virtual environment
echo "[1/5] Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "      Virtual environment activated"
else
    echo "      ERROR: venv directory not found!"
    echo "      Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if Ollama is running
echo ""
echo "[2/5] Checking Ollama status..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "      Ollama is running"
else
    echo "      WARNING: Ollama doesn't appear to be running!"
    echo "      Start it with: ollama serve"
fi

# List available Ollama models
echo ""
echo "[3/5] Available Ollama models:"
if command -v ollama &> /dev/null; then
    ollama list 2>/dev/null | head -20 || echo "      Could not fetch model list"
else
    echo "      Ollama CLI not found in PATH"
fi

# Run migrations (in case there are pending ones)
echo ""
echo "[4/5] Checking database migrations..."
python manage.py migrate --check > /dev/null 2>&1 || python manage.py migrate

# Kill any existing processes on our ports
cleanup() {
    echo ""
    echo "Shutting down..."
    pkill -f "python manage.py qcluster" 2>/dev/null || true
    pkill -f "python manage.py runserver" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start Django-Q cluster in background
echo ""
echo "[5/5] Starting services..."
echo "      Starting Django-Q worker..."
python manage.py qcluster > /dev/null 2>&1 &
QCLUSTER_PID=$!
sleep 2

if ps -p $QCLUSTER_PID > /dev/null 2>&1; then
    echo "      Django-Q worker started (PID: $QCLUSTER_PID)"
else
    echo "      WARNING: Django-Q worker may have failed to start"
fi

# Start Django development server
echo "      Starting Django server on http://0.0.0.0:8000 ..."
echo ""
echo "=========================================="
echo "  Ready! Open http://localhost:8000"
echo "  Press Ctrl+C to stop all services"
echo "=========================================="
echo ""

python manage.py runserver 0.0.0.0:8000
