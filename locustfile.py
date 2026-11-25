"""
Load test for the PhytoGuard AI API.

Each simulated user uploads leaf images from data/test for prediction, and
occasionally checks /health, like a real user of the dashboard.

Example (30 users, 5 new users per second, 2 minutes, results saved as CSV):
    locust -f locustfile.py --host http://localhost:8000 --headless -u 30 -r 5 -t 2m --csv results/containers_1
"""

import glob
import os
import random

from locust import HttpUser, between, task

IMAGE_PATHS = glob.glob(os.path.join("data", "test", "*", "*.jpg"))
if not IMAGE_PATHS:
    raise SystemExit("No images found in data/test. Run Locust from the project root folder.")

# Load a fixed sample of images into memory once, so disk reads do not affect timings
random.seed(42)
SAMPLE_IMAGES = []
for path in random.sample(IMAGE_PATHS, min(50, len(IMAGE_PATHS))):
    with open(path, "rb") as f:
        SAMPLE_IMAGES.append((os.path.basename(path), f.read()))


class PlantDiseaseUser(HttpUser):
    # Each user waits 1 to 2 seconds between requests
    wait_time = between(1, 2)

    @task(5)
    def predict(self):
        name, image = random.choice(SAMPLE_IMAGES)
        with self.client.post(
            "/predict",
            files={"file": (name, image, "image/jpeg")},
            name="POST /predict",
            catch_response=True,
        ) as response:
            if response.status_code != 200 or "predicted_class" not in response.text:
                response.failure(f"Unexpected response: {response.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
