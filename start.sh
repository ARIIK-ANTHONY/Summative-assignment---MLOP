#!/bin/sh
# Runs the FastAPI backend (internal, port 8000) and the Streamlit UI
# (public, port 8501) in the same container. streamlit_app.py already
# defaults API_BASE_URL to http://localhost:8000, so no env var is
# needed when both run together like this.
set -e

uvicorn api:app --host 0.0.0.0 --port 8000 &

exec streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
