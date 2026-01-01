# app.py
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # важливо для Streamlit Cloud (Python 3.13)

import streamlit as st
from PIL import Image

from model import get_model, predict_topk, MODEL_NAME

st.set_page_config(
    page_title="Розпізнавання зображень (Keras Applications)",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
<style>
.big-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0.2rem;}
.subtle {opacity: 0.8; margin-top: 0;}
.card {padding: 1rem; border-radius: 16px; border: 1px solid rgba(120,120,120,0.25);}
.small {font-size: 0.92rem; opacity: 0.85;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🧠 Розпізнавання зображень (Keras Applications)</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtle">Завантаж фото або зроби знімок камерою → натисни <b>«Розпізнати»</b> → отримай Top‑K результатів.</p>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ Про модель / що використано", expanded=False):
    st.write(f"**Модель:** `{MODEL_NAME}` з `keras.applications` (попередньо навчена на ImageNet).")
    st.caption("Запуск без TensorFlow: використовується Keras 3 + Torch backend (сумісно з Python 3.13 у Streamlit Cloud).")

# --------- Кешуємо модель ----------
@st.cache_resource(show_spinner=False)
def load_cached():
    return get_model()

model = load_cached()

# --------- Ввід зображення ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📤 Завантажити файл", "📷 Камера"])

image: Image.Image | None = None

with tab1:
    uploaded = st.file_uploader("Оберіть JPG/PNG", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)

with tab2:
    cam = st.camera_input("Зробіть фото")
    if cam:
        image = Image.open(cam)

st.markdown("</div>", unsafe_allow_html=True)

if image is not None:
    st.image(image, caption="Вхідне зображення", use_container_width=True)

colA, colB = st.columns([1, 1])
with colA:
    top_k = st.slider("Скільки результатів показувати (Top‑K)", 1, 10, 5)
with colB:
    show_probs = st.checkbox("Показувати ймовірності (%)", value=True)

run = st.button("🔍 Розпізнати", type="primary", use_container_width=True, disabled=(image is None))

if run and image is not None:
    with st.spinner("Класифікую зображення…"):
        preds = predict_topk(image, model, top_k=top_k)

    st.subheader("✅ Результати")
    for i, p in enumerate(preds, start=1):
        st.write(f"**{i}. {p['label_ua']}**  \n`{p['label_en']}`")
        if show_probs:
            st.progress(p["score"])
            st.caption(f"Ймовірність: **{p['score']*100:.2f}%**")

    st.divider()
    st.caption("Порада: для кращих результатів використовуйте чіткі фото об’єкта по центру кадру.")
else:
    st.info("Завантажте зображення або скористайтесь камерою, потім натисніть **«Розпізнати»**.")
