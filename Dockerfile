 # ── Stage 1: Clone MiroFish + apply patches + build frontend ────────────────
  FROM node:20 AS frontend-builder

  RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates python3 \
      && rm -rf /var/lib/apt/lists/*

  WORKDIR /app

  RUN git clone --depth=1 https://github.com/666ghj/MiroFish.git mirofish

  COPY patches/frontend/src/views/BettingView.vue mirofish/frontend/src/views/BettingView.vue
  COPY patches/inject_router.py ./
  RUN python3 inject_router.py

  WORKDIR /app/mirofish/frontend
  RUN npm install && npm run build

  # ── Stage 2: Python backend ──────────────────────────────────────────────────
  FROM python:3.11-slim

  WORKDIR /app

  RUN apt-get update && apt-get install -y --no-install-recommends \
      libxml2-dev libxslt-dev gcc git curl ca-certificates \
      && rm -rf /var/lib/apt/lists/*

  RUN pip install --no-cache-dir uv

  RUN git clone --depth=1 https://github.com/666ghj/MiroFish.git mirofish

  COPY patches/backend/app/config.py                             mirofish/backend/app/config.py
  COPY patches/backend/app/services/local_zep.py                 mirofish/backend/app/services/local_zep.py
  COPY patches/backend/app/services/zep_entity_reader.py         mirofish/backend/app/services/zep_entity_reader.py
  COPY patches/backend/app/services/zep_graph_memory_updater.py  mirofish/backend/app/services/zep_graph_memory_updater.py
  COPY patches/backend/app/services/zep_tools.py                 mirofish/backend/app/services/zep_tools.py
  COPY patches/backend/requirements.txt                          mirofish/backend/requirements.txt

  COPY patches/backend/requirements.txt requirements.txt
  RUN uv pip install --system --no-cache -r requirements.txt
  RUN uv pip install --system --no-cache -r mirofish/backend/requirements.txt || true

  COPY football/  ./football/
  COPY data/      ./data/
  COPY seed/      ./seed/
  COPY predictor/ ./predictor/
  COPY api_server.py ./

  COPY --from=frontend-builder /app/mirofish/frontend/dist ./frontend_dist

  RUN mkdir -p data

  ENV PYTHONUNBUFFERED=1
  ENV PORT=8080
  ENV FLASK_ENV=production
  ENV PYTHONPATH=/app/mirofish/backend

  EXPOSE 8080

  COPY docker-start.sh ./
  RUN chmod +x docker-start.sh
  CMD ["./docker-start.sh"]
