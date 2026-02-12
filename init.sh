#!/bin/bash
#
# The Council - Initialization Script
#
# Usage:
#   ./init.sh          Start via systemd services (survives SSH disconnects)
#   ./init.sh --dev    Start in foreground (for development/debugging)
#   ./init.sh --stop   Stop systemd services
#   ./init.sh --status Show service status
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =========================================================================
# Shared helpers
# =========================================================================

check_venv() {
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "      Virtual environment activated"
    else
        echo "      ERROR: venv directory not found!"
        echo "      Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
}

check_ollama() {
    echo ""
    echo "[2/5] Checking Ollama status..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "      Ollama is running"
    else
        echo "      WARNING: Ollama doesn't appear to be running!"
        echo "      Start it with: ollama serve"
    fi
}

list_models() {
    echo ""
    echo "[3/5] Available Ollama models:"
    if command -v ollama &> /dev/null; then
        ollama list 2>/dev/null | head -20 || echo "      Could not fetch model list"
    else
        echo "      Ollama CLI not found in PATH"
    fi
}

run_migrations() {
    echo ""
    echo "[4/5] Checking database migrations..."
    python manage.py migrate --check > /dev/null 2>&1 || python manage.py migrate
}

# =========================================================================
# Systemd service management (default mode)
# =========================================================================

install_services() {
    echo "      Installing systemd services..."
    sudo cp "$SCRIPT_DIR/council-web.service" /etc/systemd/system/
    sudo cp "$SCRIPT_DIR/council-worker.service" /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable council-web council-worker
    echo "      Services installed and enabled"
}

start_services() {
    # Install/update service files if they changed
    if ! diff -q "$SCRIPT_DIR/council-web.service" /etc/systemd/system/council-web.service > /dev/null 2>&1 || \
       ! diff -q "$SCRIPT_DIR/council-worker.service" /etc/systemd/system/council-worker.service > /dev/null 2>&1; then
        install_services
    fi

    echo ""
    echo "[5/5] Starting services..."
    echo "      Starting Django-Q worker (systemd)..."
    sudo systemctl restart council-worker
    sleep 2
    if systemctl is-active --quiet council-worker; then
        echo "      Django-Q worker is running"
    else
        echo "      WARNING: Django-Q worker may have failed to start"
        echo "      Check logs with: sudo journalctl -u council-worker -f"
    fi

    echo "      Starting Django web server (systemd)..."
    sudo systemctl restart council-web
    sleep 2
    if systemctl is-active --quiet council-web; then
        echo "      Django web server is running"
    else
        echo "      WARNING: Django web server may have failed to start"
        echo "      Check logs with: sudo journalctl -u council-web -f"
    fi

    echo ""
    echo "=========================================="
    echo "  Ready! Open http://$(hostname -I | awk '{print $1}'):8000"
    echo ""
    echo "  Services will survive SSH disconnects"
    echo "  and laptop sleep."
    echo ""
    echo "  Useful commands:"
    echo "    View logs:     sudo journalctl -u council-web -u council-worker -f"
    echo "    Stop services: ./init.sh --stop"
    echo "    Status:        ./init.sh --status"
    echo "    Dev mode:      ./init.sh --dev"
    echo "=========================================="
}

stop_services() {
    echo "Stopping services..."
    sudo systemctl stop council-web council-worker 2>/dev/null || true
    echo "Services stopped."
}

show_status() {
    echo ""
    echo "=== Council Web Server ==="
    systemctl status council-web --no-pager 2>/dev/null || echo "  Not installed"
    echo ""
    echo "=== Council Worker ==="
    systemctl status council-worker --no-pager 2>/dev/null || echo "  Not installed"
    echo ""
    echo "=== Ollama ==="
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  Ollama is running"
    else
        echo "  Ollama is NOT running"
    fi
}

# =========================================================================
# Development mode (foreground, tied to terminal - old behavior)
# =========================================================================

run_dev() {
    set -e

    cleanup() {
        echo ""
        echo "Shutting down..."
        pkill -f "python manage.py qcluster" 2>/dev/null || true
        pkill -f "python manage.py runserver" 2>/dev/null || true
        exit 0
    }
    trap cleanup SIGINT SIGTERM SIGHUP

    # Start Django-Q cluster in background
    echo ""
    echo "[5/5] Starting services (dev mode - foreground)..."
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
    echo ""
    echo "  WARNING: Dev mode - processes will stop"
    echo "  if this terminal session ends."
    echo "  Use ./init.sh (without --dev) for"
    echo "  persistent background services."
    echo "=========================================="
    echo ""

    python manage.py runserver 0.0.0.0:8000
}

# =========================================================================
# Main entry point
# =========================================================================

echo "=========================================="
echo "  The Council - Multi-Model LLM Voting"
echo "=========================================="
echo ""

case "${1:-}" in
    --stop)
        stop_services
        exit 0
        ;;
    --status)
        show_status
        exit 0
        ;;
    --dev)
        echo "[1/5] Activating virtual environment..."
        check_venv
        check_ollama
        list_models
        run_migrations
        run_dev
        ;;
    *)
        # Default: systemd service mode
        echo "[1/5] Activating virtual environment..."
        check_venv
        check_ollama
        list_models
        run_migrations
        start_services
        ;;
esac
