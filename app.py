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
from PIL import Image, ImageDraw
 
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
DEFAULT_FRCNN_PATH = os.path.join(APP_DIR, "fasterrcnn.pth")
 
 
# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLOv8 detector...")
def load_yolo(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)
 
 
def _secret(name):
    val = os.environ.get(name)
    if val:
        return val
    if hasattr(st, "secrets"):
        try:
            return st.secrets.get(name)
        except Exception:
            return None
    return None
 
 
def resolve_file_path(local_path: str, key_prefix: str) -> str:
    """
    Resolve any weights/data file, in this order:
    1) Use it if it already exists locally (e.g. committed to the repo,
       or dropped next to app.py).
    2) <PREFIX>_HF_REPO / <PREFIX>_HF_FILE (env var or st.secrets) ->
       download from the Hugging Face Hub.
    3) <PREFIX>_GDRIVE_URL (env var or st.secrets) -> download from
       Google Drive via gdown (handles the large-file confirm page).
    Keeping files out of git avoids GitHub's 100MB push limit and
    "file too big to preview" / LFS pointer issues entirely.
    """
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
 
    repo_id = _secret(f"{key_prefix}_HF_REPO")
    filename = _secret(f"{key_prefix}_HF_FILE")
    if repo_id and filename:
        from huggingface_hub import hf_hub_download
        with st.spinner(f"Downloading {os.path.basename(local_path)} from {repo_id} ..."):
            return hf_hub_download(repo_id=repo_id, filename=filename)
 
    gdrive_url = _secret(f"{key_prefix}_GDRIVE_URL")
    if gdrive_url:
        import gdown
        with st.spinner(f"Downloading {os.path.basename(local_path)} from Google Drive ..."):
            gdown.download(url=gdrive_url, output=local_path, quiet=False)
        return local_path
 
    return local_path  # will not exist -> caller shows a clear error
 
 
def resolve_fasterrcnn_path(local_path: str) -> str:
    return resolve_file_path(local_path, "FRCNN")
 
 
@st.cache_resource(show_spinner="Loading Faster R-CNN detector...")
def load_fasterrcnn(weights_path: str, num_classes: int = 2):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device
 
 
def run_fasterrcnn_inference(model, device, image: Image.Image, conf_threshold: float):
    import torch
    import torchvision.transforms.functional as F
 
    tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
 
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    keep = scores >= conf_threshold
    return boxes[keep], scores[keep]
 
 
def draw_boxes(image: Image.Image, boxes: np.ndarray, scores: np.ndarray) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill="lime")
    return annotated
 
 
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
 
st.sidebar.subheader("Detector")
detector_choice = st.sidebar.radio(
    "Model to use",
    ["YOLOv8", "Faster R-CNN"],
    help="Only the selected model is loaded into memory — the two are "
         "not loaded at once, to stay within free-tier RAM limits.",
)
 
st.sidebar.subheader("Model files")
model_file = st.sidebar.file_uploader("YOLOv8 weights (best.pt)", type=["pt"])
frcnn_file = st.sidebar.file_uploader("Faster R-CNN weights (.pth)", type=["pth"])
rules_file = st.sidebar.file_uploader("Association rules (rules.pkl)", type=["pkl"])
 
model_path = DEFAULT_MODEL_PATH
if model_file is not None:
    model_path = os.path.join(APP_DIR, "uploaded_best.pt")
    with open(model_path, "wb") as f:
        f.write(model_file.getbuffer())
 
frcnn_path = DEFAULT_FRCNN_PATH
if frcnn_file is not None:
    frcnn_path = os.path.join(APP_DIR, "uploaded_fasterrcnn.pth")
    with open(frcnn_path, "wb") as f:
        f.write(frcnn_file.getbuffer())
 
rules_path = DEFAULT_RULES_PATH
if rules_file is not None:
    rules_path = os.path.join(APP_DIR, "uploaded_rules.pkl")
    with open(rules_path, "wb") as f:
        f.write(rules_file.getbuffer())
 
st.sidebar.subheader("Detection settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider(
    "IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05,
    help="Only used by YOLOv8 — Faster R-CNN applies its own internal NMS.",
)
 
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
resolved_rules_path = resolve_file_path(rules_path, "RULES")
if not os.path.exists(resolved_rules_path):
    st.error(
        f"Rules file not found at `{rules_path}`. Upload rules.pkl from the sidebar, "
        "or set RULES_HF_REPO / RULES_HF_FILE, or RULES_GDRIVE_URL (env var or "
        "st.secrets) to fetch it automatically at startup."
    )
    st.stop()
 
rules_df = load_rules(resolved_rules_path)
all_products = get_all_products(rules_df)
 
yolo_model = None
frcnn_model = None
frcnn_device = None
 
if detector_choice == "YOLOv8":
    resolved_model_path = resolve_file_path(model_path, "YOLO")
    if not os.path.exists(resolved_model_path):
        st.error(
            f"YOLO weights not found at `{model_path}`. Upload best.pt from the sidebar, "
            "or set YOLO_HF_REPO / YOLO_HF_FILE, or YOLO_GDRIVE_URL (env var or "
            "st.secrets) to fetch it automatically at startup."
        )
        st.stop()
    yolo_model = load_yolo(resolved_model_path)
else:
    resolved_frcnn_path = resolve_fasterrcnn_path(frcnn_path)
    if not os.path.exists(resolved_frcnn_path):
        st.error(
            f"Faster R-CNN weights not found at `{frcnn_path}`. Upload the .pth file "
            "from the sidebar, or set FRCNN_HF_REPO / FRCNN_HF_FILE, or FRCNN_GDRIVE_URL "
            "(env var or st.secrets) to fetch it automatically at startup."
        )
        st.stop()
    frcnn_model, frcnn_device = load_fasterrcnn(resolved_frcnn_path)
 
 
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
 
        with st.spinner(f"Running detection ({detector_choice})..."):
            if detector_choice == "YOLOv8":
                results = yolo_model.predict(
                    np.array(image),
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                result = results[0]
                boxes_obj = result.boxes
                num_detections = len(boxes_obj) if boxes_obj is not None else 0
                confidences = boxes_obj.conf.cpu().numpy() if num_detections else np.array([])
 
                annotated = result.plot()  # BGR numpy array with boxes drawn
                annotated_rgb = annotated[:, :, ::-1]
            else:
                box_coords, confidences = run_fasterrcnn_inference(
                    frcnn_model, frcnn_device, image, conf_threshold
                )
                num_detections = len(box_coords)
                annotated_rgb = np.array(draw_boxes(image, box_coords, confidences))
 
        with col_img:
            st.image(annotated_rgb, caption=f"Detections ({detector_choice})", use_container_width=True)
 
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
 
- **Detection** — a YOLOv8n model and/or a Faster R-CNN (ResNet50-FPN)
  model, both trained on **SKU-110K** to locate products on shelf images
  (single class: `object`). Pick one from the sidebar — only that model is
  loaded into memory at a time.
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
 
