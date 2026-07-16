# RetailBrain AI — Streamlit App

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

`best.pt` and `rules.pkl` are already in this folder, so the app picks them
up automatically. You can also swap them from the sidebar without touching
the code.

## What it does
1. **Shelf Detection & Analytics tab** — upload a shelf photo, runs the
   YOLOv8 detector, draws boxes, shows count / confidence stats and a
   simple restock-status heuristic.
2. **Recommendations tab** — pick the products you know are on/off the
   shelf, get cross-sell / restock suggestions from the association-rule
   model (`rules.pkl`), with the supporting rules shown underneath.

## Known limitation
The uploaded YOLOv8 model was trained on SKU-110K with a single class
(`object`) — it detects *that* something is a product, not *which* product
it is. That's why product selection for the recommendation tab is manual
for now; wiring in a classifier/product-recognition step is the natural
next step.
