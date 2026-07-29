"""
locustfile.py

Flood-request simulation against the classifier API's /predict endpoint,
using real sample images from data/test/. Run this against the API while
varying the number of running containers to compare latency/response time.

Usage:
    locust -f locustfile.py --host http://localhost:8000
Then open http://localhost:8089, set number of users + spawn rate, and
start the test. Export the results (Download Data tab) for the README's
"Results from Flood Request Simulation" section.
"""
import os
import random

from locust import HttpUser, between, task

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(BASE_DIR, "data", "test")


def _load_sample_images():
    samples = []
    for cls in os.listdir(TEST_DIR):
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            continue
        fpath = os.path.join(cls_dir, files[0])
        with open(fpath, "rb") as f:
            samples.append((os.path.basename(fpath), f.read()))
    return samples


SAMPLE_IMAGES = _load_sample_images()


class WildlifeClassifierUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def predict(self):
        filename, data = random.choice(SAMPLE_IMAGES)
        self.client.post(
            "/predict",
            files={"file": (filename, data, "image/jpeg")},
            name="/predict",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
