#!/bin/bash
set -e

echo "============================================"
echo "  AstroLens v1.0.0"
echo "  Galaxy Anomaly Discovery System"
echo "============================================"
echo ""
echo "  API:  http://localhost:8000"
echo "  Web:  http://localhost:8080"
echo "  Docs: http://localhost:8000/docs"
echo ""

# Start API server in background
echo "Starting API server on port 8000..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "API is ready."
        break
    fi
    sleep 1
done

# Start Web UI
echo "Starting Web UI on port 8080..."
ASTROLENS_API=http://localhost:8000 python -m web.app --host 0.0.0.0 --port 8080 &
WEB_PID=$!

echo ""
echo "Both services are running."
echo "Press Ctrl+C to stop."

# Wait for either process to exit
wait -n $API_PID $WEB_PID
exit $?
