# African Wildlife Classifier

Image classification pipeline for African wildlife (buffalo, elephant, rhino, zebra) using MobileNetV2 transfer learning, served via a FastAPI backend and a Streamlit UI, with retraining and load-testing support.

## Video Demo

**YouTube:** _TODO - add link_ (demo of prediction + upload/retrain, camera on)

## Deployed URLs

- **API:** https://summative-assignment-mlop-0kui.onrender.com (Swagger docs at `/docs`)
- **UI:** _TODO - add public URL once deployed_

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

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Project_name
```

### 2. Install dependencies
Requires Python 3.11+ (the saved `.keras` model files need Keras 3.15+, which requires Python >= 3.11).
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Dataset
The `data/train/` and `data/test/` folders are already included, sorted by class. If you need to regenerate them from scratch, the notebook's Step 2 handles it with `kagglehub` (`kagglehub.dataset_download("biancaferreira/african-wildlife")` — handles Kaggle auth and caching on its own, no manual API key setup), then Step 3 runs the conversion logic in `src/preprocessing.py` (`convert_yolo_to_classification`), which walks the extracted YOLO-format images/labels, does its own 85/15 train/test split per class, and writes out the `data/train/<class>/` and `data/test/<class>/` folders.

### 4. Train / retrain the model
Open and run `notebook/project_name.ipynb` top to bottom (runs locally end to end; a GPU speeds it up but CPU works fine too, just slower). It walks through:
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
```bash
uvicorn api:app --reload --port 8000
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
In a second terminal (API must be running):
```bash
streamlit run streamlit_app.py
```
Opens at http://localhost:8501 with three tabs: **Predict** (upload an image, click Predict), **Data Insights** (dataset visualizations), and **Upload & Retrain** (bulk upload + retraining trigger + status). The API URL is set via the `API_BASE_URL` environment variable (defaults to `http://localhost:8000`; point it at the deployed API's public URL once hosted).

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
This repo includes a [`render.yaml`](render.yaml) Blueprint that deploys both services straight from the existing Dockerfiles — no local Docker install needed, Render builds the images on their end.

1. Push this repo to GitHub (already done if you're reading this on GitHub).
2. On [render.com](https://render.com), sign up/log in, then **New > Blueprint**, and connect this GitHub repo. Render reads `render.yaml` and provisions two free web services: `african-wildlife-api` (from `Dockerfile`) and `african-wildlife-ui` (from `Dockerfile.streamlit`).
3. Once `african-wildlife-api` finishes its first deploy, copy its assigned URL (Render shows it on the service page, e.g. `https://african-wildlife-api-xxxx.onrender.com`). If it doesn't exactly match the `API_BASE_URL` value already set in `render.yaml`, update that env var on the `african-wildlife-ui` service to the real URL and let it redeploy.
4. Open the `african-wildlife-ui` service's URL — that's the live UI, talking to the live API.

**Free-tier caveats worth knowing before you demo:** free Render web services spin down after ~15 minutes idle (the next request wakes it back up but takes 30-60s), and free services don't have a persistent disk — a model retrained live will work for that session but won't survive the service restarting/redeploying. Neither of these affects the retraining/prediction *demonstration* itself, they're just not durable across restarts on the free tier.

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
