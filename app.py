# app.py
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Keras 3 + Torch backend (сумісно зі Streamlit Cloud / Python 3.13)

import streamlit as st
from PIL import Image
import pandas as pd

from model import get_model, predict_topk, MODEL_NAME

st.set_page_config(
    page_title="Розпізнавання зображень (Keras Applications)",
    page_icon="🧠",
    layout="centered",
)

# -------------------- Стилі --------------------
st.markdown(
    """
<style>
.big-title {font-size: 2.2rem; font-weight: 850; margin: 0.2rem 0 0.3rem 0;}
.subtle {opacity: 0.85; margin: 0 0 0.8rem 0; font-size: 1.0rem;}
.card {padding: 1rem; border-radius: 18px; border: 1px solid rgba(120,120,120,0.25); background: rgba(255,255,255,0.02);}
.badge {display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px; border: 1px solid rgba(120,120,120,0.25); font-size: 0.85rem; opacity: 0.9;}
.small {font-size: 0.93rem; opacity: 0.85;}
hr {margin: 1.1rem 0;}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Хедер --------------------
st.markdown('<div class="big-title">🧠 Розпізнавання зображень</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle">Завантаж фото або зроби знімок камерою → натисни <b>«Розпізнати»</b> → отримай Top‑K результатів.</div>',
    unsafe_allow_html=True,
)
st.markdown(f'<span class="badge">Модель: {MODEL_NAME}</span>', unsafe_allow_html=True)

# -------------------- Сайдбар: опис моделі --------------------
with st.sidebar:
    st.header("ℹ️ Про модель")
    st.write(f"**{MODEL_NAME}** — попередньо навчена модель з `keras.applications`, навчена на датасеті **ImageNet**.")
    st.markdown(
        """
**Що вона вміє розпізнавати?**  
ImageNet містить **~1000 класів**: тварини (кіти/собаки/птахи), транспорт (авто/літаки/кораблі), предмети побуту, їжа, інструменти та інші типові об'єкти.

**Як читати результат?**  
Модель повертає Top‑K гіпотез з ймовірностями. Це *не 100% гарантія*, а оцінка впевненості моделі.

**Обмеження:**  
- Найкраще працює на чітких фото з одним об’єктом.  
- Може плутати схожі класи (напр., породи собак).  
- Не призначена для облич/емоцій/персонажів конкретних серіалів без донавчання.
"""
    )
    st.divider()
    st.caption("Технічна примітка: Keras 3 працює з Torch backend, тому додаток стабільно деплоїться у Streamlit Cloud на Python 3.13.")

# -------------------- Кеш: модель --------------------
@st.cache_resource(show_spinner=False)
def load_cached():
    return get_model()

model = load_cached()

# -------------------- Ввід зображення --------------------
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
    st.image(image, caption="Вхідне зображення", use_column_width=True)

# -------------------- Налаштування --------------------
colA, colB, colC = st.columns([1, 1, 1])
with colA:
    top_k = st.slider("Top‑K", 1, 10, 5)
with colB:
    show_probs = st.checkbox("Показувати %", value=True)
with colC:
    show_chart = st.checkbox("Показати графік", value=True)

# -------------------- Кнопка --------------------
run = st.button("🔍 Розпізнати", type="primary", use_container_width=True, disabled=(image is None))

if run and image is not None:
    with st.spinner("Класифікую зображення…"):
        preds = predict_topk(image, model, top_k=top_k)

    # Top-1 виділення
    top1 = preds[0]
    st.success(f"Найімовірніше: **{top1['label_ua']}**  ·  `{top1['label_en']}`")

    st.subheader("✅ Top‑K результати")
    for i, p in enumerate(preds, start=1):
        st.write(f"**{i}. {p['label_ua']}**  \n`{p['label_en']}`")
        if show_probs:
            st.progress(p["score"])
            st.caption(f"Ймовірність: **{p['score']*100:.2f}%**")

    if show_chart:
        st.subheader("📊 Візуалізація впевненості")
        df = pd.DataFrame(
            {
                "Клас": [p["label_ua"] for p in preds],
                "Ймовірність": [p["score"] for p in preds],
            }
        ).set_index("Клас")
        st.bar_chart(df)

    st.divider()
    st.markdown(
        """
**Поради для кращого розпізнавання:**
- роби фото з хорошим освітленням;
- бажано, щоб **один об’єкт** був у центрі кадру;
- уникай сильного розмиття та дуже темних знімків.
"""
    )
else:
    st.info("Завантажте зображення або скористайтесь камерою, потім натисніть **«Розпізнати»**.")
