"""
PhytoGuard AI - Streamlit dashboard.

Run locally (with the API already running on port 8000):
    streamlit run app/streamlit_app.py

Set API_URL to point the dashboard at a deployed API, for example:
    set API_URL=https://phytoguard-api.onrender.com
"""

import json
import os
import time

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
EDA_PATH = os.path.join(os.path.dirname(__file__), "eda_summary.json")

GREEN = "#2e8b57"
BROWN = "#a0522d"
STATUS_COLOURS = alt.Scale(domain=["Healthy", "Diseased"], range=[GREEN, BROWN])

st.set_page_config(page_title="PhytoGuard AI", page_icon="🌿", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path, timeout=75):
    try:
        response = requests.get(f"{API_URL}{path}", timeout=timeout)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as error:
        return None, str(error)


def api_post(path, timeout=120, **kwargs):
    try:
        response = requests.post(f"{API_URL}{path}", timeout=timeout, **kwargs)
        data = response.json()
        if response.status_code >= 400:
            return None, data.get("detail", f"Error {response.status_code}")
        return data, None
    except requests.RequestException as error:
        return None, str(error)
    except ValueError:
        return None, "The API returned an invalid response."


@st.cache_data
def load_eda():
    with open(EDA_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_classes():
    data, _ = api_get("/classes")
    return data["classes"] if data else []


def readable(class_name):
    crop, _, condition = class_name.partition("___")
    return f"{crop.replace('_', ' ').replace(',', '')}: {condition.replace('_', ' ')}"


def percent(value):
    return f"{value * 100:.2f}%" if isinstance(value, (int, float)) else "n/a"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🌿 PhytoGuard AI")
st.sidebar.caption("Plant disease detection from leaf images")

page = st.sidebar.radio(
    "Go to",
    ["Model Status", "Predict", "Data Visualisations", "Upload & Retrain"],
)

with st.spinner("Connecting to the API. On the free cloud plan it can take up to a minute to wake up..."):
    health, health_error = api_get("/health")
if health:
    st.sidebar.success(f"API online · model {health['model_version']}")
    st.sidebar.caption(f"Up-time: {health['uptime_readable']}")
else:
    st.sidebar.error("API offline")
    st.sidebar.caption(API_URL)

st.sidebar.divider()
st.sidebar.caption(f"API: {API_URL}")


# ---------------------------------------------------------------------------
# Page 1: Model Status
# ---------------------------------------------------------------------------

def page_status():
    st.title("Model Status")
    st.caption("Live up-time and performance of the deployed model.")

    if st.button("🔄 Refresh"):
        st.rerun()

    if not health:
        st.error(f"Cannot reach the API at {API_URL}. Is it running?\n\n{health_error}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", health["status"].upper())
    c2.metric("Up-time", health["uptime_readable"])
    c3.metric("Model version", health["model_version"])
    c4.metric("Retraining", health["retraining"].capitalize())
    st.caption(f"Server started at {health['started_at']} (UTC) · container: {health['container_id']}")

    metrics, _ = api_get("/metrics")
    if not metrics:
        return

    model = metrics["model"]
    st.subheader("Test-set performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", percent(model.get("test_accuracy")))
    m2.metric("Precision (weighted)", percent(model.get("weighted_precision")))
    m3.metric("Recall (weighted)", percent(model.get("weighted_recall")))
    m4.metric("F1 (weighted)", percent(model.get("weighted_f1")))
    m5.metric("F1 (macro)", percent(model.get("macro_f1")))
    st.caption(
        f"Last trained: {model.get('trained_on', 'n/a')}. "
        "Accuracy and loss were measured on 5,431 unseen PlantVillage test images. "
        + (model.get("note", ""))
    )

    st.subheader("Live request statistics")
    r1, r2 = st.columns(2)
    r1.metric("Predictions served", metrics["requests"]["predictions_served"])
    r2.metric("Average prediction latency", f"{metrics['requests']['average_latency_ms']} ms")
    st.caption("Counted since the server started.")

    st.subheader("Retraining history")
    history = metrics["retrain_history"]
    if history:
        df = pd.DataFrame(history)
        cols = [c for c in ["timestamp", "method", "new_images", "replay_images",
                            "old_test_accuracy", "new_test_accuracy", "promoted",
                            "evaluation_set"] if c in df.columns]
        st.dataframe(df[cols].iloc[::-1], hide_index=True, width="stretch")
    else:
        st.info("No retraining runs yet.")


# ---------------------------------------------------------------------------
# Page 2: Predict
# ---------------------------------------------------------------------------

def page_predict():
    st.title("Predict a Leaf Image")
    st.caption("Upload one photo of a plant leaf (JPEG or PNG). The model identifies the crop and its condition.")

    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Tip: try an image from the data/test folder of the repository.")
        return

    left, right = st.columns([1, 1.3])
    left.image(uploaded, caption=uploaded.name, width="stretch")

    with right:
        with st.spinner("Analysing the leaf..."):
            result, error = api_post(
                "/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
            )

        if error:
            st.error(error)
            return

        if result["is_healthy"]:
            st.success(f"### ✅ {result['label']}")
        else:
            st.error(f"### ⚠️ {result['label']}")

        st.metric("Confidence", f"{result['confidence']:.2f}%")
        st.caption(f"Crop: {result['crop']} · Condition: {result['condition']} · "
                   f"Response time: {result['latency_ms']} ms")

        if result["confidence"] < 70:
            st.warning("The model is not very confident. Check the other likely classes below, "
                       "and consider taking a clearer photo of a single leaf on a plain background.")

        top = pd.DataFrame(result["top_predictions"])
        chart = alt.Chart(top).mark_bar(color=GREEN).encode(
            x=alt.X("confidence:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("label:N", sort="-x", title=None),
            tooltip=["label", alt.Tooltip("confidence:Q", format=".2f")],
        ).properties(height=160, title="Top 3 predictions")
        st.altair_chart(chart, width="stretch")


# ---------------------------------------------------------------------------
# Page 3: Data Visualisations
# ---------------------------------------------------------------------------

def page_visualisations():
    st.title("Data Visualisations")
    eda = load_eda()
    st.caption(f"Source: {eda['source']}.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Crops & health status", "2. Leaf colour", "3. Leaf texture", "Model errors",
    ])

    # Feature 1
    with tab1:
        counts = pd.DataFrame(
            [{"class": k, "images": v} for k, v in eda["class_counts"].items()]
        )
        counts["crop"] = counts["class"].str.split("___").str[0].str.replace("_", " ").str.replace(",", "")
        counts["status"] = counts["class"].str.contains("healthy").map({True: "Healthy", False: "Diseased"})
        by_crop = counts.groupby(["crop", "status"], as_index=False)["images"].sum()
        order = by_crop.groupby("crop")["images"].sum().sort_values(ascending=False).index.tolist()

        chart = alt.Chart(by_crop).mark_bar().encode(
            x=alt.X("images:Q", title="Number of images"),
            y=alt.Y("crop:N", sort=order, title=None),
            color=alt.Color("status:N", scale=STATUS_COLOURS, title="Status"),
            tooltip=["crop", "status", "images"],
        ).properties(height=420, title="Images per crop by health status")
        st.altair_chart(chart, width="stretch")

        total = eda["total_images"]
        tomato = by_crop[by_crop["crop"] == "Tomato"]["images"].sum()
        diseased = counts[counts["status"] == "Diseased"]["images"].sum()
        a, b, c = st.columns(3)
        a.metric("Total images", f"{total:,}")
        b.metric("Tomato share", f"{tomato / total * 100:.1f}%")
        c.metric("Diseased share", f"{diseased / total * 100:.1f}%")

        st.markdown(
            "**What it tells us:** Tomato makes up a third of the dataset and has 10 of the 38 classes, "
            "and 72% of images are diseased. Some crops only have one health status: Orange and Squash "
            "are only diseased, while Soybean, Blueberry and Raspberry are only healthy. The model can "
            "never learn what a healthy orange leaf looks like, which is a gap that new uploads and "
            "retraining can fill."
        )

    feat = pd.DataFrame(eda["feature_by_crop"])

    # Feature 2
    with tab2:
        overall = eda["feature_overall"]
        a, b = st.columns(2)
        a.metric("Green pixels: healthy vs diseased",
                 f"{overall['Healthy']['green_ratio'] * 100:.1f}% vs {overall['Diseased']['green_ratio'] * 100:.1f}%")
        b.metric("Yellow/brown pixels: healthy vs diseased",
                 f"{overall['Healthy']['yellow_brown_ratio'] * 100:.2f}% vs {overall['Diseased']['yellow_brown_ratio'] * 100:.2f}%")

        for column, title in [("green_ratio", "Share of green pixels per crop"),
                              ("yellow_brown_ratio", "Share of yellow/brown pixels per crop")]:
            chart = alt.Chart(feat).mark_bar().encode(
                x=alt.X("status:N", title=None, sort=["Healthy", "Diseased"], axis=alt.Axis(labels=False, ticks=False)),
                y=alt.Y(f"{column}:Q", title="Share of pixels", axis=alt.Axis(format="%")),
                color=alt.Color("status:N", scale=STATUS_COLOURS, title="Status"),
                column=alt.Column("crop:N", title=None, header=alt.Header(labelOrient="bottom")),
                tooltip=["crop", "status", alt.Tooltip(f"{column}:Q", format=".2%")],
            ).properties(width=55, height=220, title=title)
            st.altair_chart(chart)

        st.markdown(
            "**What it tells us:** colour is the clearest visible sign of disease. Diseased leaves have "
            "more yellow or brown pixels than healthy leaves in **all nine crops**, and less green in "
            "seven of nine. Diseased tomato leaves show only 1.78% yellow/brown, because most tomato "
            "diseases appear as small dark spots. This weak colour signal helps explain why tomato "
            "diseases are the hardest classes for the model."
        )
        st.caption(eda["feature_sample"])

    # Feature 3
    with tab3:
        overall = eda["feature_overall"]
        st.metric("Average texture score: healthy vs diseased",
                  f"{overall['Healthy']['texture']:.4f} vs {overall['Diseased']['texture']:.4f}")
        chart = alt.Chart(feat).mark_bar().encode(
            x=alt.X("status:N", title=None, sort=["Healthy", "Diseased"], axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("texture:Q", title="Average edge strength"),
            color=alt.Color("status:N", scale=STATUS_COLOURS, title="Status"),
            column=alt.Column("crop:N", title=None, header=alt.Header(labelOrient="bottom")),
            tooltip=["crop", "status", alt.Tooltip("texture:Q", format=".4f")],
        ).properties(width=55, height=220, title="Texture score per crop")
        st.altair_chart(chart)

        st.markdown(
            "**What it tells us:** whole-image texture does **not** separate healthy and diseased leaves. "
            "The averages are almost equal, and the direction flips from crop to crop: diseased leaves "
            "have more texture in Grape, Pepper and Corn, but less in Tomato, Potato, Cherry and Peach. "
            "Veins, leaf edges and backgrounds add as many edges as disease spots. The model has to learn "
            "where texture appears and what shape it has, which is why a CNN is needed."
        )

    # Model errors
    with tab4:
        ev = eda["evaluation"]
        a, b, c = st.columns(3)
        a.metric("Test accuracy", percent(ev["test_accuracy"]))
        b.metric("Macro F1", percent(ev["macro_f1"]))
        c.metric("Misclassified", f"{ev['misclassified']} of {ev['test_images']:,}")

        weak = pd.DataFrame(ev["weakest_classes"])
        weak["label"] = weak["class"].map(readable)
        weak_long = weak.melt(id_vars="label", value_vars=["precision", "recall", "f1"],
                              var_name="metric", value_name="score")
        chart = alt.Chart(weak_long).mark_bar().encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
            y=alt.Y("label:N", sort=weak["label"].tolist(), title=None),
            color=alt.Color("metric:N", title="Metric"),
            yOffset="metric:N",
            tooltip=["label", "metric", alt.Tooltip("score:Q", format=".2%")],
        ).properties(height=520, title="The 10 weakest classes on the test set")
        st.altair_chart(chart, width="stretch")

        st.markdown(
            "**What it tells us:** seven of the ten weakest classes are tomato diseases. Tomato Early "
            "blight has 98% precision but only 57% recall, so the model misses many Early blight leaves, "
            "mostly confusing them with Late blight and Septoria leaf spot. These are the classes where "
            "uploading more images and retraining will help most."
        )


# ---------------------------------------------------------------------------
# Page 4: Upload & Retrain
# ---------------------------------------------------------------------------

def page_retrain():
    st.title("Upload Data & Retrain")
    st.caption("Add new labelled leaf images, then retrain the model on them.")

    classes = load_classes()
    if not classes:
        st.error(f"Cannot reach the API at {API_URL}.")
        return

    status, _ = api_get("/retrain/status")
    if status:
        pending, threshold = status["pending_uploads"], status["retrain_threshold"]
        st.subheader("New data waiting for retraining")
        st.progress(min(pending / threshold, 1.0),
                    text=f"{pending} of {threshold} images (automatic retraining starts at {threshold})")

    st.subheader("1. Upload labelled images")
    mode = st.radio("Upload method", ["Images for one class", "Zip file with class folders"], horizontal=True)

    if mode == "Images for one class":
        class_name = st.selectbox("Class of these images", classes, format_func=readable)
        files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if st.button("Upload images", disabled=not files, type="primary"):
            with st.spinner(f"Uploading {len(files)} images..."):
                result, error = api_post(
                    "/upload",
                    data={"class_name": class_name},
                    files=[("files", (f.name, f.getvalue(), f.type)) for f in files],
                )
            show_upload_result(result, error)
    else:
        st.markdown(
            "The zip must contain one folder per class, named exactly like the class, for example "
            "`Tomato___Early_blight/leaf1.jpg`. The `data/retrain` folder of the repository has this layout."
        )
        archive = st.file_uploader("Choose a .zip file", type=["zip"])
        if st.button("Upload zip", disabled=archive is None, type="primary"):
            with st.spinner("Uploading and checking images..."):
                result, error = api_post(
                    "/upload/zip",
                    timeout=300,
                    files={"file": (archive.name, archive.getvalue(), "application/zip")},
                )
            show_upload_result(result, error)

    st.subheader("2. Retrain the model")
    st.markdown(
        "Retraining updates the model's classification layer using the uploaded images plus a replay "
        "sample of the original training data, on top of the frozen MobileNetV2 feature extractor. "
        "Both models are then tested on the same held-out images, and the new model only replaces "
        "the current one if it is at least as accurate."
    )

    if st.button("🚀 Start retraining", type="primary"):
        result, error = api_post("/retrain")
        if error:
            st.error(error)
        else:
            st.success(f"Retraining started on {result['new_images']} new images.")
            time.sleep(1)
            st.rerun()

    show_retrain_status()


def show_upload_result(result, error):
    if error:
        st.error(error)
        return
    st.success(f"Saved {result['saved']} images. {result['pending_uploads']} images are now waiting for retraining.")
    rejected = result.get("rejected_count", len(result.get("rejected", [])))
    if rejected:
        st.warning(f"{rejected} files were rejected (unknown class folder or not a valid image).")
    if result.get("auto_retrain_started"):
        st.info(f"Automatic retraining started: {result['reason']}")


def show_retrain_status():
    status, _ = api_get("/retrain/status")
    if not status or status["status"] == "idle":
        return

    st.subheader("Retraining status")
    if status["status"] == "running":
        st.info(f"⏳ Retraining is running (trigger: {status['trigger']}, started {status['started_at']} UTC). "
                "This usually takes under a minute.")
        if st.button("Check progress"):
            st.rerun()
        return

    if status["status"] == "failed":
        st.error(f"Retraining failed: {status['error']}")
        return

    result = status["result"] or {}
    if not result.get("started"):
        st.info(result.get("reason", "Retraining did not start."))
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy before", percent(result["old_test_accuracy"]))
    c2.metric("Accuracy after", percent(result["new_test_accuracy"]),
              delta=f"{(result['new_test_accuracy'] - result['old_test_accuracy']) * 100:+.2f} pts")
    c3.metric("Images used", f"{result['new_images']} new + {result['replay_images']} replay")

    if result["promoted"]:
        st.success("✅ The retrained model was better or equal, so it is now the live model.")
    else:
        st.warning("The retrained model was less accurate, so the previous model was kept.")
    st.caption(f"Evaluated on {result['evaluation_set']} · finished {status['finished_at']} UTC · "
               f"method: {result.get('method', 'fine-tuning')}")


# ---------------------------------------------------------------------------

PAGES = {
    "Model Status": page_status,
    "Predict": page_predict,
    "Data Visualisations": page_visualisations,
    "Upload & Retrain": page_retrain,
}
PAGES[page]()
