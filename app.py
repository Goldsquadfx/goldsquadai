import streamlit as st
from PIL import Image
from transformers import pipeline
import random

st.set_page_config(page_title="GoldSquadFx AI Analyzer", layout="centered")
st.set_page_config(page_title="", layout="centered")
st.title("📈 GoldSquadFx AI Analyzer")
st.write("Upload a trading chart image to get an instant trade suggestion.")

uploaded = st.file_uploader("Upload your chart (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Chart", use_column_width=True)
    st.write("🔍 Analyzing your chart...")

    classifier = pipeline("image-classification", model="kyujinpy/chart-pattern-recognition")
    results = classifier(image)
    label = results[0]["label"].lower()
    conf = round(results[0]["score"] * 100, 2)
except Exception:
    # fallback trend detector
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
