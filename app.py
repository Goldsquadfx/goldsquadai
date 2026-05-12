import streamlit as st
from PIL import Image
from transformers import pipeline
import cv2
import numpy as np
import random

# --- Page setup ---
st.set_page_config(page_title="GoldSquadFX AI Analyzer", layout="centered")
st.title("📈 GoldSquadFX – AI Chart Analyzer")
st.write("Upload a trading chart image to get an instant trade suggestion.")

# --- File uploader ---
uploaded = st.file_uploader("Upload your chart (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Chart", use_column_width=True)
    st.write("🔍 Analyzing your chart...")

    # --- Try chart-specific model ---
    try:
        classifier = pipeline("image-classification", model="kyujinpy/chart-pattern-recognition")
        results = classifier(image)
        label = results[0]["label"].lower()
        conf = round(results[0]["score"] * 100, 2)

        if any(x in label for x in ["up", "bull", "rising", "ascending", "breakout", "uptrend"]):
            direction = "BUY"
        elif any(x in label for x in ["down", "bear", "falling", "descending", "downtrend"]):
            direction = "SELL"
        else:
            direction = "WAIT / NEUTRAL"

    except Exception:
        # --- Fallback: simple image-based slope detector (no ML) ---
        img = np.array(image.convert("L"))
        edges = cv2.Canny(img, 50, 150)
        gradient = np.mean(np.gradient(edges.astype(float)))
        label = "fallback"
        conf = 0
        if gradient > 0.05:
            direction = "BUY"
        elif gradient < -0.05:
            direction = "SELL"
        else:
            direction = "WAIT / NEUTRAL"

    # --- Generate pseudo price levels for example output ---
    current_price = random.uniform(1.2000, 1.3000)
    if direction == "BUY":
        tp = current_price * 1.005   # +0.5%
        sl = current_price * 0.9975  # -0.25%
    elif direction == "SELL":
        tp = current_price * 0.995
        sl = current_price * 1.0025
    else:
        tp = sl = current_price

    # --- Display result ---
    st.markdown(f"""
    ### 🧭 Trade Suggestion
    **Signal:** {direction}  
    **Confidence:** {conf}%  
    **Entry:** {current_price:.4f}  
    **Take Profit:** {tp:.4f}  
    **Stop Loss:** {sl:.4f}  
    **Detected Pattern:** *{label}*
    """)
