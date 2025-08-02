import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set page
st.set_page_config(page_title="The great toast evaluation lab", page_icon="🍞", layout="centered")

# Custom CSS: background + glass box
st.markdown("""
    <style>
    .stApp {
        background-image: 
            linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
            url('https://github.com/angelinarose-2025/ECG-CLASSIFIER/blob/main/original-672b1f646f4bc35e1966f5daba769d5f.gif?raw=true');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .glass-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.25);
        max-width: 850px;
        margin: 40px auto;
    }

    .glass-box h2, .glass-box p, .glass-box label {
        color: #fff;
        text-align: center;
    }

    .burn-box {
        background: #ffffff11;
        border-left: 8px solid #fff;
        color: #fff;
        padding: 20px;
        border-radius: 15px;
        margin-top: 30px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        animation: slideIn 0.5s ease;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-15px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Burn level styles
burn_styles = {
    "🟡 Uncooked": {"color": "#ffeb3b", "emoji": "🟡"},
    "🟠 Lightly Toasted": {"color": "#ff9800", "emoji": "🟠"},
    "🟤 Medium Toasted": {"color": "#6d4c41", "emoji": "🟤"},
    "⚫ Dark Toasted": {"color": "#37474f", "emoji": "⚫"},
    "🔥 is that food or....": {"color": "#f44336", "emoji": "🔥"},
}

# Brightness calculation
def get_masked_brightness(image, patch_size=100):
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2
    half = patch_size // 2
    top = max(cy - half, 0)
    bottom = min(cy + half, h)
    left = max(cx - half, 0)
    right = min(cx + half, w)
    patch = image[top:bottom, left:right]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = gray < 230
    valid_pixels = gray[mask]
    return valid_pixels.mean() if len(valid_pixels) else gray.mean()

# Burn classifier
def classify_burn(image):
    brightness = get_masked_brightness(image)
    if brightness > 180:
        return "🟡 Uncooked"
    elif brightness > 140:
        return "🟠 Lightly Toasted"
    elif brightness > 100:
        return "🟤 Medium Toasted"
    elif brightness > 60:
        return "⚫ Dark Toasted"
    else:
        return "🔥 Burnt to a Crisp!"

# ⬛️ Begin main UI in glass box
st.markdown('<div class="glass-box">', unsafe_allow_html=True)

st.markdown("## 🍞 The Great Toast Evaluation Lab")
st.markdown("Snap your slice and let's see if it's edible or evidence 🔍")

uploaded_file = st.file_uploader("Upload toast image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(img, caption="Your uploaded toast", use_column_width=True)

    # Get burn level
    burn_level = classify_burn(img_cv)
    style = burn_styles[burn_level]

    # Show animated burn level box
    st.markdown(f"""
        <div class="burn-box" style="border-left-color: {style['color']}; color: {style['color']}">
            {style['emoji']} Burn Level: <span style="color:{style['color']};">{burn_level}</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

