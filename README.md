# PhytoGuard AI: Plant Disease Detection Pipeline

PhytoGuard AI detects plant diseases from a photo of a leaf. It covers the full machine learning cycle: data acquisition, preprocessing, model training and evaluation, a prediction API, a web dashboard, bulk data upload, retraining with an automatic trigger, Docker containers, load testing and cloud deployment.

| | Link |
|---|---|
| **Web dashboard** | https://phytoguard-ui.onrender.com |
| **API (Swagger docs)** | https://phytoguard-api.onrender.com/docs |
| **Video demo** | ADD_YOUR_YOUTUBE_LINK_HERE |

> **Note:** the app runs on Render's free plan, which sleeps after 15 minutes without visits. The first visit can take about a minute while it wakes up. After that it responds normally.

## 1. Project Description

Crop diseases cause major losses for smallholder farmers, and many farmers cannot easily reach an agricultural expert to identify a disease in time. This project builds on my Introduction to Machine Learning summative, [Credit Card Payment Default Prediction](https://github.com/kauriane1/creditDefaultProject), where I compared traditional machine learning and deep learning models on tabular data (30,000 customers, an imbalanced binary classification task). This assignment requires non-tabular data, and credit records do not come as images or sound, so I kept the same end-to-end classification workflow and applied it to an image-based problem: detecting plant diseases from leaf photos. The evaluation approach carries over directly: stratified data splits, macro-F1 alongside accuracy, per-class precision and recall, confusion matrices, and a close look at the classes the model gets wrong.

PhytoGuard AI classifies a leaf image into one of **38 classes** covering **14 crops** (including tomato, potato, maize, apple, grape and pepper), either as healthy or as a specific disease. A farmer or extension officer uploads a photo and receives the crop, the condition, a confidence score and the top three most likely classes.

**Dataset:** [PlantVillage](https://www.tensorflow.org/datasets/catalog/plant_village) (54,303 leaf images, 38 classes), loaded through TensorFlow Datasets and split 80% train, 10% validation and 10% test.

**Model:** MobileNetV2 pretrained on ImageNet, used for transfer learning and then fine-tuned.

| Metric (test set, 5,431 unseen images) | Score |
|---|---|
| Accuracy | **96.11%** (96.32% after the first retraining) |
| Weighted precision / recall / F1 | 96.20% / 96.11% / 96.04% |
| Macro F1 (every class weighted equally) | 95.07% |
| Test loss | 0.1262 (0.1237 after retraining) |


## 2. Features

| Requirement | Where to find it |
|---|---|
| **Model prediction** from one image | Dashboard → *Predict*, or `POST /predict` |
| **Visualisations** with interpretations of 3 features | Dashboard → *Data Visualisations*, and notebook Section 2.6 |
| **Bulk upload** of new data (many images, or a zip of class folders) | Dashboard → *Upload & Retrain*, or `POST /upload` and `POST /upload/zip` |
| **Retraining trigger**: retrain button plus automatic trigger at 200 new images | Dashboard → *Upload & Retrain*, or `POST /retrain` |
| **Model up-time** and monitoring | Dashboard → *Model Status*, or `GET /health` and `GET /metrics` |
| **Evaluation in production** | Each retraining compares the old and new model on held-out images, and the new one is only promoted if it is at least as accurate |
| **Load testing** with different numbers of Docker containers | Section 7 below, and the `results/` folder |


## 3. Architecture

```
                 ┌──────────────────────────┐
  User  ───────► │  Streamlit dashboard     │  Model status, predict,
                 │  (phytoguard-ui)         │  visualisations, upload, retrain
                 └────────────┬─────────────┘
                              │ HTTP
                 ┌────────────▼─────────────┐
  Locust ──────► │  FastAPI service         │  /predict  /upload  /retrain
                 │  (phytoguard-api)        │  /health   /metrics
                 └────────────┬─────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
  MobileNetV2 feature   Classification head    data/train, data/test,
  extractor (.tflite)   (.npz) + class names   data/uploads (new images)
```

Locally, `docker-compose` runs the dashboard, an **nginx load balancer** and any number of API containers behind it.


## 4. Repository Structure

```
PhytoGuard-AI/
├── README.md
├── notebook/
│   └── phytoguard_ai.ipynb        # Full pipeline: EDA, preprocessing, training, evaluation, retraining
├── src/
│   ├── config.py                  # Paths and settings shared by every module
│   ├── preprocessing.py           # Image decoding, resizing, normalisation, loading class folders
│   ├── model.py                   # Model building (Keras), export for serving (TFLite)
│   ├── prediction.py              # Single-image prediction with top-3 results
│   ├── retrain.py                 # Retraining trigger, retraining, comparison and promotion
│   └── api.py                     # FastAPI service
├── app/
│   ├── streamlit_app.py           # Web dashboard
│   └── eda_summary.json           # Dataset insights shown on the dashboard
├── data/
│   ├── train/                     # 760 images (20 per class), replay data for retraining
│   ├── test/                      # 380 images (10 per class), held-out evaluation in production
│   └── retrain/                   # 570 images (15 per class), sample bulk upload for demos
├── models/
│   ├── phytoguard_ai_final.h5     # Trained Keras model
│   ├── phytoguard_feature_extractor.tflite
│   ├── phytoguard_head.npz
│   ├── class_names.json
│   ├── metrics.json
│   └── retrain_log.json
├── results/                       # Locust load test results
├── locustfile.py
├── Dockerfile, Dockerfile.ui, docker-compose.yml, nginx.conf
├── render.yaml                    # Render deployment (Blueprint)
└── requirements.txt, requirements-ui.txt, requirements-train.txt
```

## 5. The Machine Learning Pipeline

All steps are documented in detail, with interpretations, in `notebook/phytoguard_ai.ipynb`.

1. **Data acquisition:** PlantVillage is downloaded with TensorFlow Datasets. Sample images are exported into `data/train`, `data/test` and `data/retrain` in one-folder-per-class format, which is also the format for user uploads.
2. **Data processing:** images are resized from 256 × 256 to 224 × 224 and normalised with MobileNetV2 preprocessing (pixel values scaled to the range −1 to 1). Training images are augmented with random flips, rotations and zoom.
3. **Exploratory analysis and feature interpretation:** the notebook and the dashboard examine three features of the data:
   - **Crop and health-status distribution:** tomato makes up 33% of the dataset and 72% of images are diseased. Some crops only have one health status, which the model cannot learn around.
   - **Leaf colour:** diseased leaves have more yellow or brown pixels than healthy leaves in all nine crops that have both statuses. Tomato diseases show the weakest colour signal, which matches where the model makes most mistakes.
   - **Leaf texture:** whole-image edge strength does *not* separate healthy and diseased leaves, which shows why a CNN that learns local patterns is needed.
4. **Model creation:** MobileNetV2 with a frozen base and a new classification head (global average pooling, dropout 0.2, 38-class softmax), trained for 7 epochs. The top 30 layers are then unfrozen and fine-tuned for 3 epochs with a learning rate of 1e-5.
5. **Model testing:** evaluated on 5,431 unseen test images with accuracy, loss, precision, recall, F1 (weighted and macro), a per-class report, a confusion matrix and an analysis of the most common mistakes. The weakest class is Tomato Early blight (recall 56.7%), mostly confused with Late blight and Septoria leaf spot.
6. **Model retraining:** new labelled images are combined with replay images from the original training data, the model is retrained, and both models are evaluated on the same held-out images. The new model only replaces the current one if it is at least as accurate. The old version is archived and every run is logged in `models/retrain_log.json`.
7. **API:** a FastAPI service exposes prediction, upload, retraining and monitoring endpoints.

### How retraining works in production
The notebook (Section 6) retrains by **fine-tuning the top MobileNetV2 layers** with TensorFlow. In the deployed app, retraining updates the **final classification layer** on top of the frozen MobileNetV2 features. Features of the stored images are cached, so only new uploads pass through the network. This makes retraining take seconds instead of minutes and fits within a small cloud instance, while keeping the same safety checks: replay data, comparison on held-out images, and promotion only if accuracy does not drop.

### Serving the model efficiently
For deployment, the trained Keras model (`phytoguard_ai_final.h5`) is split into a **TensorFlow Lite feature extractor** and the **classification head weights** (`python -m src.model --export`). Predictions are identical to the Keras model (maximum probability difference of 0.000003, with 100% agreement on the predicted class), but the API needs about 190 MB of memory instead of about 730 MB with full TensorFlow.

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status, model version, up-time, retraining state, container ID |
| GET | `/metrics` | Test metrics, retraining history, live request count and average latency |
| GET | `/classes` | The 38 class names |
| GET | `/dataset/stats` | Image counts per class in each data folder |
| POST | `/predict` | Upload one image and get the prediction with the top 3 classes |
| POST | `/upload` | Upload several images for one class |
| POST | `/upload/zip` | Upload a zip with one folder per class |
| POST | `/retrain` | Start retraining on the uploaded images (manual trigger) |
| GET | `/retrain/status` | Progress and results of the latest retraining |

Interactive documentation: https://phytoguard-api.onrender.com/docs


## 7. Load Testing Results (Locust)

The API was load tested with Locust on the same workload each time: **30 concurrent users**, spawned at 5 per second, for **2 minutes**. Each user uploaded leaf images from `data/test` to `POST /predict` and occasionally checked `GET /health`, waiting 1 to 2 seconds between requests. Each API container was limited to **1 CPU** to mimic a small cloud instance, and nginx distributed requests across the containers.

| POST /predict | 1 container | 2 containers | 3 containers |
|---|---|---|---|
| Requests served | 630 | 1,360 | 1,719 |
| Average response time | 2,968 ms | 582 ms | 164 ms |
| Median response time | 2,900 ms | 480 ms | 130 ms |
| 95th percentile | 4,300 ms | 1,400 ms | 330 ms |
| Throughput | 5.4 req/s | 11.5 req/s | 14.6 req/s |
| Failures | 0 | 0 | 0 |

**What the results show:**

- **One container was overloaded.** A single CPU can process about 5 to 6 predictions per second, but 30 users sent requests faster than that. Requests queued up, so the median response time rose to 2.9 seconds even though one prediction takes under 200 ms.
- **Adding containers removed the queue.** With 2 containers, throughput doubled and the median response time fell to 480 ms. With 3 containers, the median dropped to 130 ms, which is close to the time of a single prediction with no waiting.
- **The gain from 2 to 3 containers is smaller in throughput because the system caught up with demand.** With 30 users each waiting 1 to 2 seconds between requests, the maximum demand is roughly 18 requests per second. At 3 containers the API served almost all of it, so throughput was limited by the users rather than the server.
- **No request failed in any test**, so the API stayed stable even when it was overloaded. It got slower, but it did not crash.

Overall, the model scales horizontally: response time and capacity improve almost in proportion to the number of containers until the system meets demand. All tests ran on one laptop, with Locust sharing the same machine, so absolute numbers would differ on cloud servers, but the trend would be the same. These tests were run with the full TensorFlow version of the API, before the switch to TensorFlow Lite for deployment.

Raw results: `results/containers_1_stats.csv`, `containers_2_stats.csv` and `containers_3_stats.csv`.


## 8. Setup Instructions

### Option A: run with Docker (recommended)
Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).

```bash
git clone https://github.com/kauriane1/PhytoGuard-AI.git
cd PhytoGuard-AI
docker compose up --build
```

- Dashboard: http://localhost:8501
- API docs (through the nginx load balancer): http://localhost:8000/docs

Run several API containers behind the load balancer:
```bash
docker compose down
docker compose up -d --scale api=3
```

### Option B: run with Python
Requires Python 3.10 to 3.13.

```bash
git clone https://github.com/kauriane1/PhytoGuard-AI.git
cd PhytoGuard-AI
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-ui.txt

# Terminal 1: the API
uvicorn src.api:app --reload

# Terminal 2: the dashboard
streamlit run app/streamlit_app.py
```

Predict from the command line:
```bash
python -m src.prediction data/test/Tomato___Early_blight/img_000.jpg
```

### Retraining from the command line
```bash
python -m src.retrain --data data/retrain --manual
```

### Training and exporting the model
The notebook was run on Kaggle with GPU. To retrain the full Keras model locally and export the serving files:
```bash
pip install -r requirements-train.txt
python -m src.model --export
```

### Running the load test
```bash
pip install locust
locust -f locustfile.py --host http://localhost:8000 --headless -u 30 -r 5 -t 2m --csv results/containers_1
```
Or run `locust -f locustfile.py --host http://localhost:8000` and open http://localhost:8089 for live charts.

### Deploying to Render
The repository includes a Render Blueprint (`render.yaml`) that creates two free web services:
1. In Render, choose **New → Blueprint** and select this repository.
2. Set `API_URL` to the public URL of the `phytoguard-api` service.
3. Deploy. Render builds both Docker images from the repository.


## 9. Limitations and Future Work

- **Field photos:** PlantVillage images are taken on plain backgrounds under controlled light. A real outdoor photo tested in the notebook was misclassified (Early blight predicted as Late blight), which shows a domain shift. Collecting field photos through the upload feature and retraining on them is the most valuable next step.
- **Weak classes:** Tomato Early blight, Corn Cercospora leaf spot and Tomato Target Spot have the lowest F1-scores and should be prioritised when collecting new data.
- **Free hosting:** the free plan has no persistent storage, so uploaded images and retrained versions are reset when the service restarts. A paid plan with a persistent disk, or cloud storage such as S3, would keep them.
- **Single-crop coverage gaps:** some crops only have healthy or only diseased images, so the model cannot distinguish health status for them.
