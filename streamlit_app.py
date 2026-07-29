"""
streamlit_app.py

UI for the African Wildlife classifier, talking to the FastAPI service
(api.py) over HTTP so the UI and API can be deployed/scaled independently.
The API base URL is configurable via the API_BASE_URL environment variable
(defaults to localhost for local development).

Covers: prediction, dataset visualizations, and bulk upload + retrain-trigger
functionality.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

api_base_url = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ["buffalo", "elephant", "rhino", "zebra"]

st.set_page_config(page_title="African Wildlife Classifier", page_icon="🦓", layout="wide")

st.title("African Wildlife Classifier")

tab_predict, tab_insights, tab_retrain = st.tabs(
    ["Predict", "Data Insights", "Upload & Retrain"]
)

with tab_predict:
    st.write(
        "Upload a photo of a buffalo, elephant, rhino, or zebra, then click "
        "Predict."
    )
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="predict_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", width=350)

        if st.button("Predict", type="primary"):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            with st.spinner("Predicting..."):
                try:
                    response = requests.post(f"{api_base_url}/predict", files=files, timeout=30)
                    response.raise_for_status()
                except requests.RequestException as exc:
                    st.error(f"Could not reach the API at {api_base_url}: {exc}")
                else:
                    result = response.json()
                    if result.get("low_confidence"):
                        st.subheader("Prediction: Unrecognized")
                        st.warning(
                            "⚠️ This doesn't confidently match any of the 4 known "
                            "classes (buffalo, elephant, rhino, zebra). The model "
                            "always has to name one of them internally — its "
                            f"closest guess was **{result['predicted_class']}** at "
                            f"only {result['confidence']:.2%} confidence, which is "
                            "too low to trust."
                        )
                    else:
                        st.subheader(f"Prediction: {result['predicted_class']}")
                        st.write(f"Confidence: {result['confidence']:.2%}")
                    st.write("Class probabilities:")
                    st.bar_chart(result["probabilities"])

with tab_insights:
    st.write(
        "Interpretations of the dataset behind the model — computed from "
        "the images in `data/train/` and `data/test/`."
    )

    @st.cache_data(show_spinner="Scanning dataset images...")
    def compute_dataset_stats():
        rows = []
        for split in ["train", "test"]:
            split_dir = os.path.join(BASE_DIR, "data", split)
            for cls in CLASS_NAMES:
                cls_dir = os.path.join(split_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    fpath = os.path.join(cls_dir, fname)
                    try:
                        with Image.open(fpath) as img:
                            width, height = img.size
                            brightness = float(np.array(img.convert("L")).mean())
                    except (OSError, ValueError):
                        continue
                    rows.append({
                        "split": split,
                        "class": cls,
                        "aspect_ratio": width / height,
                        "brightness": brightness,
                    })
        return pd.DataFrame(rows)

    stats_df = compute_dataset_stats()

    if stats_df.empty:
        st.warning("No images found under data/train or data/test.")
    else:
        st.subheader("1. Class distribution (train vs test)")
        counts = stats_df.groupby(["class", "split"]).size().unstack(fill_value=0)
        st.bar_chart(counts)
        st.caption(
            "Classes are balanced across both splits, so the model isn't "
            "biased toward any single animal just because it saw more "
            "examples of it."
        )

        st.subheader("2. Average brightness by class")
        st.bar_chart(stats_df.groupby("class")["brightness"].mean())
        st.caption(
            "Brightness is similar across classes, meaning the model can't "
            "shortcut classification by learning 'darker photo = rhino' "
            "instead of real visual features."
        )

        st.subheader("3. Average aspect ratio by class")
        st.bar_chart(stats_df.groupby("class")["aspect_ratio"].mean())
        st.caption(
            "Aspect ratios are close across classes, so resizing every "
            "image to a fixed 224x224 square doesn't distort one animal's "
            "shape more than another's."
        )

with tab_retrain:
    st.subheader("1. Upload bulk images for retraining")
    class_label = st.selectbox("Class label for these images", CLASS_NAMES)
    bulk_files = st.file_uploader(
        "Choose one or more images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="bulk_uploader",
    )
    if st.button("Upload for retraining"):
        if not bulk_files:
            st.warning("Choose at least one image first.")
        else:
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type)) for f in bulk_files
            ]
            try:
                resp = requests.post(
                    f"{api_base_url}/upload",
                    data={"class_label": class_label},
                    files=files_payload,
                    timeout=60,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")
            else:
                saved = resp.json()["saved"]
                st.success(f"Uploaded {saved} image(s) to class '{class_label}'.")

    st.divider()
    st.subheader("2. Trigger retraining")
    epochs = st.number_input("Epochs", min_value=1, max_value=50, value=5)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start retraining", type="primary"):
            try:
                resp = requests.post(
                    f"{api_base_url}/retrain", params={"epochs": int(epochs)}, timeout=10
                )
                resp.raise_for_status()
                st.info(resp.json()["message"])
            except requests.RequestException as exc:
                st.error(f"Could not start retraining: {exc}")
    with col2:
        if st.button("Check status"):
            st.session_state["watch_retrain"] = True

    if st.session_state.get("watch_retrain"):
        try:
            status = requests.get(f"{api_base_url}/retrain/status", timeout=5).json()
            st.json(status)
            if status.get("status") == "running":
                st.caption("Auto-refreshing every 3s while retraining runs...")
                time.sleep(3)
                st.rerun()
            else:
                st.session_state["watch_retrain"] = False
        except requests.RequestException as exc:
            st.error(f"Could not fetch status: {exc}")
            st.session_state["watch_retrain"] = False
