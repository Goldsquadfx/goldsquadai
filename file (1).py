import streamlit as st
from PIL import Image
from transformers import pipeline
import random

st.set_page_config(page_title="Big Snapper AI Analyzer", layout="centered")
st.title("📈 Big Snapper – AI Chart Analyzer")
st.write("Upload a trading chart image to get an instant trade suggestion.")

uploaded = st.file_uploader("Upload your chart (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Chart", use_column_width=True)
    st.write("🔍 Analyzing your chart...")

    classifier = pipeline("image-classification", model="microsoft/resnet-50")
    result = classifier(image)[0]
    label = result["label"].lower()
    conf = round(result["score"] * 100, 2)

    if any(x in label for x in ["up", "bull", "rising", "ascending"]):
        direction = "BUY"
    elif any(x in label for x in ["down", "bear", "falling", "descending"]):
        direction = "SELL"
    else:
        direction = "WAIT / NEUTRAL"

    price = random.uniform(1.2000, 1.3000)
    if direction == "BUY":
        tp, sl = price * 1.005, price * 0.9975
    elif direction == "SELL":
        tp, sl = price * 0.995, price * 1.0025
    else:
        tp = sl = price

    st.markdown(f"""
    ### 🧭 Trade Suggestion
    **Signal:** {direction}  
    **Confidence:** {conf}%  
    **Entry:** {price:.4f}  
    **Take Profit:** {tp:.4f}  
    **Stop Loss:** {sl:.4f}  
    **Pattern:** *{label}*
    """)
