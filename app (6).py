"""
RetailBrain AI - Streamlit App
--------------------------------
Retail shelf product detection (YOLOv8) + Market-basket recommendations.

Run:
    streamlit run app.py

Expects best.pt and rules.pkl in the same folder as this file
(override with the sidebar file uploaders if needed).
"""

import os
import io
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RetailBrain AI",
    page_icon="🛒",
    layout="wide",
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(APP_DIR, "best.pt")
DEFAULT_RULES_PATH = os.path.join(APP_DIR, "rules.pkl")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 detector...")
def load_yolo(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


@st.cache_resource(show_spinner="Loading association rules...")
def load_rules(rules_path: str):
    # rules.pkl was saved with joblib (numpy arrays are wrapped),
    # plain pickle.load fails on it -> use joblib.load first, fall back to pickle.
    try:
        return joblib.load(rules_path)
    except Exception:
        with open(rules_path, "rb") as f:
            return pickle.load(f)


def get_all_products(rules: pd.DataFrame):
    products = set()
    for col in ("antecedents", "consequents"):
        for fs in rules[col]:
            products.update(fs)
    return sorted(products)


def recommend(products, rules: pd.DataFrame, min_confidence: float = 0.0):
    """Same logic as recommend.py, plus a confidence filter and rule details."""
    matched_rows = []
    for product in products:
        rec = rules[rules["antecedents"].apply(lambda x: product in x)]
        rec = rec[rec["confidence"] >= min_confidence]
        matched_rows.append(rec)

    if not matched_rows:
        return [], pd.DataFrame()

    all_matches = pd.concat(matched_rows).drop_duplicates()

    recommendations = set()
    for _, row in all_matches.iterrows():
        for item in row["consequents"]:
            recommendations.add(item)
    recommendations -= set(products)

    detail_rows = all_matches.copy()
    detail_rows["antecedents"] = detail_rows["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    detail_rows["consequents"] = detail_rows["consequents"].apply(lambda s: ", ".join(sorted(s)))
    detail_cols = ["antecedents", "consequents", "confidence", "lift", "support"]
    detail_cols = [c for c in detail_cols if c in detail_rows.columns]
    detail_rows = detail_rows[detail_cols].sort_values("confidence", ascending=False)

    return sorted(recommendations), detail_rows


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🛒 RetailBrain AI")
st.sidebar.caption("Retail shelf analytics — detection + recommendations")

st.sidebar.subheader("Model files")
model_file = st.sidebar.file_uploader("YOLOv8 weights (best.pt)", type=["pt"])
rules_file = st.sidebar.file_uploader("Association rules (rules.pkl)", type=["pkl"])

model_path = DEFAULT_MODEL_PATH
if model_file is not None:
    model_path = os.path.join(APP_DIR, "uploaded_best.pt")
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())

rules_path = DEFAULT_RULES_PATH
if rules_file is not None:
    rules_path = os.path.join(APP_DIR, "uploaded_rules.pkl")
    with open(rules_path, "wb") as f:
        f.write(rules_file.getbuffer())

st.sidebar.subheader("Detection settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05)

st.sidebar.subheader("Recommendation settings")
min_conf_rule = st.sidebar.slider("Min. rule confidence", 0.0, 1.0, 0.0, 0.05)

st.sidebar.info(
    "ℹ️ The uploaded YOLOv8 model was trained on SKU-110K with a single "
    "class (`object`) — it locates and counts products on the shelf but "
    "does not identify *which* product each box is. Product identity for "
    "the recommendation engine is therefore selected manually below."
)

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
if not os.path.exists(model_path):
    st.error(f"YOLO weights not found at `{model_path}`. Upload best.pt from the sidebar.")
    st.stop()

if not os.path.exists(rules_path):
    st.error(f"Rules file not found at `{rules_path}`. Upload rules.pkl from the sidebar.")
    st.stop()

yolo_model = load_yolo(model_path)
rules_df = load_rules(rules_path)
all_products = get_all_products(rules_df)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛒 RetailBrain AI")
st.caption("Intelligent Retail Shelf Analytics Platform")

tab_detect, tab_reco, tab_about = st.tabs(
    ["📷 Shelf Detection & Analytics", "🔁 Recommendations", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Detection & Analytics
# ---------------------------------------------------------------------------
with tab_detect:
    st.subheader("Upload a shelf image")
    img_file = st.file_uploader("Shelf image", type=["jpg", "jpeg", "png"], key="shelf_img")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        col_img, col_stats = st.columns([2, 1])

        with st.spinner("Running detection..."):
            results = yolo_model.predict(
                np.array(image),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        result = results[0]
        boxes = result.boxes

        annotated = result.plot()  # BGR numpy array with boxes drawn
        annotated_rgb = annotated[:, :, ::-1]

        with col_img:
            st.image(annotated_rgb, caption="Detections", use_container_width=True)

        num_detections = len(boxes) if boxes is not None else 0
        confidences = boxes.conf.cpu().numpy() if num_detections else np.array([])

        with col_stats:
            st.metric("Products detected", num_detections)
            if num_detections:
                st.metric("Avg. confidence", f"{confidences.mean():.2f}")
                st.metric("Min / Max confidence", f"{confidences.min():.2f} / {confidences.max():.2f}")
            else:
                st.metric("Avg. confidence", "—")

            # Simple shelf-status heuristic based on count
            st.subheader("Shelf status")
            low_th = st.number_input("Low-stock threshold (# products)", min_value=1, value=20, step=1)
            if num_detections == 0:
                status = "⚠️ No products detected"
            elif num_detections < low_th:
                status = "🔴 Needs restocking"
            elif num_detections < low_th * 2:
                status = "🟡 Moderately stocked"
            else:
                status = "🟢 Well stocked"
            st.write(status)

        if num_detections:
            st.subheader("Confidence distribution")
            hist_df = pd.DataFrame({"confidence": confidences})
            st.bar_chart(hist_df["confidence"].round(1).value_counts().sort_index())

            det_table = pd.DataFrame({
                "detection_id": range(1, num_detections + 1),
                "confidence": confidences.round(3),
            })
            st.dataframe(det_table, use_container_width=True, hide_index=True)

            csv_bytes = det_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download detections (CSV)",
                data=csv_bytes,
                file_name="detections.csv",
                mime="text/csv",
            )
    else:
        st.info("Upload a shelf image to run detection.")

# ---------------------------------------------------------------------------
# Tab 2 — Recommendations
# ---------------------------------------------------------------------------
with tab_reco:
    st.subheader("Detected / selected products")
    st.caption(
        "Pick the products currently seen on the shelf (or out of stock) "
        "to get restocking / cross-sell recommendations from the "
        "association-rule model."
    )

    selected_products = st.multiselect(
        "Products",
        options=all_products,
        default=[],
    )

    if st.button("Get recommendations", type="primary"):
        if not selected_products:
            st.warning("Select at least one product first.")
        else:
            recs, detail_df = recommend(selected_products, rules_df, min_confidence=min_conf_rule)
            if recs:
                st.success(f"{len(recs)} recommendation(s) found")
                st.write(recs)
                st.subheader("Supporting rules")
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
            else:
                st.info("No recommendation found for this selection at the chosen confidence.")

    with st.expander("Browse all products in the rule base"):
        st.write(f"{len(all_products)} unique products")
        st.dataframe(pd.DataFrame({"product": all_products}), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3 — About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
### RetailBrain AI

An AI-powered retail shelf monitoring proof-of-concept combining:

- **Detection** — a YOLOv8n model trained on **SKU-110K** to locate products
  on shelf images (single class: `object`).
- **Recommendations** — an association-rule model (support / confidence /
  lift) trained on a separate grocery basket dataset, used to suggest
  cross-sell / restock items once product identity is known.

**Current limitation:** the detector reports *where* products are and *how
many*, but not *what* they are, so it cannot feed product names straight
into the recommender. Closing that gap (e.g. adding a classification head,
or a product-recognition model) is listed as future work in the project.

**Pipeline**

```
Upload Shelf Image
        ↓
Product Detection (YOLOv8)
        ↓
Inventory Analytics (count, confidence, status)
        ↓
Manual/assisted product selection
        ↓
Recommendation Generation (association rules)
```
        """
    )
