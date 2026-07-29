# African Wildlife Classifier

Image classification pipeline for African wildlife (buffalo, elephant, rhino, zebra) using MobileNetV2 transfer learning, served via a FastAPI backend and a Streamlit UI, with retraining and load-testing support.

**GitHub repo:** https://github.com/ARIIK-ANTHONY/Summative-assignment---MLOP

## Video Demo

**YouTube:** https://youtu.be/ZKaPQMBtc0k (demo of prediction + upload/retrain, camera on)

## Deployed URLs

- **UI:** https://african-wildlife-ui.onrender.com/ — start here, this is the app: Predict, Data Insights, and Upload & Retrain tabs
- **API:** https://summative-assignment-mlop-0kui.onrender.com/docs — interactive Swagger docs for the underlying FastAPI service (`/predict`, `/upload`, `/retrain`, `/health`)

**Note on retraining on the deployed version:** both services above run on Render's free tier (512MB RAM). Prediction and Data Insights work fine there, but a real retraining run (`model.fit()`) needs more memory than 512MB provides and will crash the free instance. Retraining is demonstrated in the video against a local run of the same code instead (see [Setup Instructions](#setup-instructions) below to run it yourself) — the trigger, upload, preprocessing, and retraining logic are identical either way, only the deployed instance's RAM is the limiting factor.

## Project Description

This project classifies images of African wildlife into 4 categories using a MobileNetV2-based transfer learning model. It's built for the MLOps summative assignment, covering data acquisition, preprocessing, model training, evaluation, a retraining pipeline, an API, a UI, and load testing.

## Dataset

**Source:** [African Wildlife (Kaggle)](https://www.kaggle.com/datasets/biancaferreira/african-wildlife), uploaded by Bianca Ferreira, 2020.

**What it is:** A dataset of 4 African animals in YOLO labeling format (jpg images paired with txt bounding-box label files), originally built to support real-time animal detection on embedded devices deployed in South African nature reserves. Images were labeled for object detection using Google's image search results.

**Classes and size:** 1,504 images total across 4 balanced classes (~376 images each): buffalo, elephant, rhino, zebra.

**Note on splits:** The raw Kaggle download is not pre-split into train/test — that split is something Ultralytics added on their own mirror of this dataset. This project's conversion script (in the notebook and `src/preprocessing.py`) does its own 85/15 train/test split per class after converting from YOLO format to classification folders.

**Known benchmarks:** Published results on this dataset range from ~67% accuracy (DenseNet-201 baseline) up to ~99% (Vision Transformer, at much higher compute cost). This project's MobileNetV2 model reaches ~94.7% accuracy on the held-out test set after fine-tuning (see [Model Evaluation Summary](#model-evaluation-summary)).

**Limitation to be aware of:** the dataset favors well-photographed, daylight conditions, so model performance may not generalize well to rarer or nocturnal sightings, or genuine unposed camera-trap footage.

**Label noise:** because the source images were labeled for object detection (one bounding box class per photo) rather than classification, a handful end up folder-labeled by whichever animal the original annotator boxed, even when a different animal is the dominant subject of the photo. For example, `data/test/zebra/2 (291).jpg` shows mostly an elephant with a zebra small in the background, and `data/test/zebra/3 (374).jpg` shows only a rhino. The model's calls on both are consistent with what's actually dominant in the frame — the "wrong" label is a byproduct of the original YOLO annotation, not a model error. Worth keeping in mind both when reading test-set accuracy and when picking a demo image.

## Repository Structure

```
Project_name/
│
├── README.md
├── requirements.txt
├── api.py                        # FastAPI service: /predict, /upload, /retrain, /health
├── streamlit_app.py              # UI: prediction, data insights, upload + retrain trigger
├── locustfile.py                 # flood-request simulation against the API
├── Dockerfile                    # containerizes the API
├── Dockerfile.streamlit          # containerizes the UI
├── docker-compose.yml            # runs API + UI together
│
├── notebook/
│   └── project_name.ipynb        # full training + evaluation notebook
│
├── src/
│   ├── preprocessing.py          # shared preprocessing / YOLO-to-classification conversion
│   ├── model.py                  # model architecture + train/retrain logic
│   └── prediction.py             # single-image inference logic
│
├── data/
│   ├── train/{buffalo,elephant,rhino,zebra}/
│   ├── test/{buffalo,elephant,rhino,zebra}/
│   └── retrain_uploads/{buffalo,elephant,rhino,zebra}/  # bulk uploads awaiting retraining (created at runtime)
│
└── models/
    ├── final_model.keras         # trained model (Keras format, equivalent to .tf SavedModel)
    ├── best_model.keras          # best checkpoint from initial training
    ├── finetuned_model.keras     # best checkpoint from fine-tuning
    └── class_indices.json        # maps class names to model output indices
```

## Setup Instructions

**Quickest path to see it working (prediction + retraining):** steps 1, 2, 6, and 7 below. Step 3 (dataset) is already done — the data is in the repo. Step 4 (notebook) is optional to re-run — it's already fully executed with saved outputs, so it can just be read. Steps 8-10 (Docker, Locust, cloud deploy) are supplementary.

### 1. Clone the repository
```bash
git clone https://github.com/ARIIK-ANTHONY/Summative-assignment---MLOP.git
cd Summative-assignment---MLOP
```

### 2. Install dependencies
Requires Python 3.11+ (the saved `.keras` model files need Keras 3.15+, which requires Python >= 3.11).
```bash
python -m venv venv          # some systems use `python3` instead of `python`
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dataset
The `data/train/` and `data/test/` folders are already included, sorted by class. If you need to regenerate them from scratch, the notebook's Step 2 handles it with `kagglehub` (`kagglehub.dataset_download("biancaferreira/african-wildlife")` — handles Kaggle auth and caching on its own, no manual API key setup), then Step 3 runs the conversion logic in `src/preprocessing.py` (`convert_yolo_to_classification`), which walks the extracted YOLO-format images/labels, does its own 85/15 train/test split per class, and writes out the `data/train/<class>/` and `data/test/<class>/` folders.

### 4. Train / retrain the model
The notebook is already fully executed with its outputs saved (preprocessing, training, evaluation, fine-tuning) — open it to review as-is, no need to re-run it:
```bash
jupyter notebook notebook/project_name.ipynb
```
If you do want to re-run it top to bottom, it works locally end to end (a GPU speeds it up, CPU works fine too, just slower). It walks through:
- loading and converting the dataset
- exploring the data (class balance, sample images, image size checks)
- preprocessing + augmentation
- building the MobileNetV2 transfer learning model
- training with Early Stopping + ModelCheckpoint
- evaluating with Accuracy, Precision, Recall, F1-score, Confusion Matrix, and Classification Report
- fine-tuning the last layers of the base model
- saving the final model + class index mapping to `models/`

Alternatively, `src/model.py` exposes a `train()` function that can be called directly (used later by the retraining trigger):
```python
from src.model import train
model, history, class_indices = train(train_dir="data/train", test_dir="data/test", epochs=10)
```

### 5. Run a prediction (direct, no server)
```python
from src.prediction import predict_image
result = predict_image("data/test/elephant/some_image.jpg")
print(result)
# {'predicted_class': 'elephant', 'confidence': 0.98, 'probabilities': {...}}
```

### 6. Run the API
Make sure the venv from step 2 is activated in this terminal (`venv\Scripts\activate` on Windows, `source venv/bin/activate` otherwise) — needed in every new terminal you open, since that's where `uvicorn`, `streamlit`, etc. are actually installed.
```bash
python -m uvicorn api:app --reload --port 8000
```
Interactive Swagger docs: http://localhost:8000/docs

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status + uptime (model up-time) |
| GET | `/classes` | list of the 4 class names |
| POST | `/predict` | upload one image → prediction + confidence + probabilities |
| POST | `/upload` | bulk-upload images (form field `class_label` + `files`) → saved to `data/retrain_uploads/<class>/` |
| POST | `/retrain` | folds uploaded images into `data/train/`, retrains starting from the current saved model (not from scratch), runs in a background thread |
| GET | `/retrain/status` | poll while retraining runs (`idle` / `running` / `done` / `failed`) |

### 7. Run the UI
In a second terminal, with the venv activated again in this new terminal (API from step 6 must still be running):
```bash
python -m streamlit run streamlit_app.py
```
Opens at http://localhost:8501 with three tabs: **Predict** (upload an image, click Predict), **Data Insights** (dataset visualizations), and **Upload & Retrain** (bulk upload + retraining trigger + status). The API URL is set via the `API_BASE_URL` environment variable (defaults to `http://localhost:8000`; the deployed UI points it at `https://summative-assignment-mlop-0kui.onrender.com`).

### 8. Run with Docker
```bash
docker compose up --build
```
Starts the API on `:8000` and the UI on `:8501`, wired together. To build/run the API image standalone:
```bash
docker build -t wildlife-api .
docker run -p 8000:8000 wildlife-api
```

### 9. Load testing with Locust
With the API running:
```bash
locust -f locustfile.py --host http://localhost:8000
```
Open http://localhost:8089, set number of users + spawn rate, and start. `locustfile.py` sends a mix of `/predict` requests (using real sample images from `data/test/`) and `/health` checks. See [Results from Flood Request Simulation](#results-from-flood-request-simulation) below.

### 10. Deploy to the cloud (Render)
This project is deployed on [Render](https://render.com) as two separate free Docker-based web services, no local Docker install needed since Render builds the images server-side:

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. **API service:** Render dashboard → **New → Web Service** → connect this repo → Dockerfile Path `./Dockerfile` → Health Check Path `/health` → Free plan → Create. This is what's live at https://summative-assignment-mlop-0kui.onrender.com.
3. **UI service:** Render dashboard → **New → Web Service** → connect this repo again → Dockerfile Path `./Dockerfile.streamlit` → Health Check Path `/` → add environment variable `API_BASE_URL` = the API service's URL from step 2 → Free plan → Create. This is what's live at https://african-wildlife-ui.onrender.com.

**Free-tier limitations found while deploying this:**
- Free Render web services spin down after ~15 minutes idle (the next request wakes them back up but takes 30-60s).
- Free services have no persistent disk — a model retrained live works for that session but won't survive a restart/redeploy.
- **512MB RAM is enough for prediction/inference but not for actually running `model.fit()`** — a live retrain attempt on the free API service OOM-crashes the container (confirmed during development; the API auto-restarts afterward, `/predict` and `/health` are unaffected). This is why the video demonstrates retraining against a local run instead — the code path is identical, only the free tier's memory ceiling is the blocker. A paid Render plan (2GB+ RAM) or a platform with more free RAM would resolve this if needed.
- This repo also has a [`render.yaml`](render.yaml) Blueprint and a [`Dockerfile.combined`](Dockerfile.combined)/[`start.sh`](start.sh) that run the API and UI together in a single container (simpler, one URL, no `API_BASE_URL` wiring needed) — tested and confirmed working locally, but it hits the same 512MB ceiling on Render's free tier since it runs TensorFlow and Streamlit in one instance instead of two. It would work as-is on any host with more free RAM per container.

## Model Evaluation Summary

See `notebook/project_name.ipynb` for the full breakdown, including the confusion matrix and per-class classification report from the original training run. The notebook reports Accuracy, Precision, Recall, F1-score, a Confusion Matrix, and a Classification Report on the held-out test set, both before and after fine-tuning.

Current `models/final_model.keras` (which has since been fine-tuned for one additional epoch via the `/retrain` endpoint) scores as follows on the held-out test set (228 images):

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| buffalo | 0.9818 | 0.9474 | 0.9643 | 57 |
| elephant | 0.9153 | 0.9474 | 0.9310 | 57 |
| rhino | 0.9123 | 0.9286 | 0.9204 | 56 |
| zebra | 0.9825 | 0.9655 | 0.9739 | 58 |

**Overall accuracy: 94.74%** (macro F1: 0.9474)

## Data Visualizations

Three data interpretations, available both in the notebook (pre-training) and live in the Streamlit UI's **Data Insights** tab:
1. **Class distribution** — image counts per class, train vs test (checks for class imbalance)
2. **Brightness by class** — average pixel intensity per class (checks whether the model could rely on lighting as a shortcut instead of real visual features)
3. **Aspect ratio by class** — confirms resizing to a fixed 224x224 square doesn't distort one class more than another

## Results from Flood Request Simulation

**Methodology note:** Docker Desktop wasn't available in the dev environment used for this run, so instead of `docker compose up --scale api=N`, each "container" below is one `uvicorn` worker process (`--workers N`) serving the exact same API code — each worker is an independent OS process with its own model instance in memory, which for this stateless, CPU-bound inference workload is the same unit of horizontal scaling that separate containers behind a load balancer would provide. All runs used `locust -f locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 40s` (20 users, spawn rate 5/s, 40s duration) sending the same real `/predict` + `/health` traffic mix from `locustfile.py` on a 12-core CPU-only machine. If you redeploy with real Docker containers or on a cloud platform, rerun this exact command against 1/2/4 containers to replace these numbers.

| Containers (worker processes) | Requests | Requests/s | Median latency (ms) | Avg latency (ms) | 95th %ile latency (ms) | Failures |
|---|---|---|---|---|---|---|
| 1 | 330 | 8.68 | 830 | 1012 | 2700 | 0 |
| 2 | 421 | 10.68 | 290 | 563 | 2200 | 0 |
| 4 | 433 | 10.99 | 220 | 524 | 1400 | 0 |

Raw Locust CSV output for all three runs is in [`locust_results/`](locust_results/) (`w1_stats.csv`, `w2_stats.csv`, `w4_stats.csv`, plus per-second history).

**What the numbers show:** going from 1 → 2 workers roughly halves median latency (830ms → 290ms) because a single Keras model instance handles predictions one at a time, so concurrent requests queue up behind it — adding a second instance lets two requests be scored in parallel. Going 2 → 4 gives a smaller further improvement in median/p95 (290ms → 220ms, p95 2200ms → 1400ms), with diminishing returns because all workers were competing for the same 12 physical cores on one machine rather than getting dedicated hardware the way separate containers on a cluster would — the p99 tail (4400ms → 6300ms) actually got noisier at 4 workers for that reason. Zero failures at every level — the API stayed correct under load, it just queued.
