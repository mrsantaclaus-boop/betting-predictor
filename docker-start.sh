#!/bin/bash
echo "==> Starting SAI Tipster API (port ${PORT:-8080})..."
cd /app && exec gunicorn -w 2 --threads 4 --timeout 180 -b "0.0.0.0:${PORT:-8080}" api_server:app
