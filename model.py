# model.py
from __future__ import annotations

import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # Keras 3 працює з backend'ом torch

from typing import List, Dict
import numpy as np
from PIL import Image

from keras.applications import mobilenet_v2
from keras.applications.imagenet_utils import decode_predictions

MODEL_NAME = "MobileNetV2 (ImageNet, Keras Applications)"

# Легкий переклад найчастіших слів (демо-словник).
# ImageNet має 1000 класів; повний переклад — це окрема велика таблиця.
_WORDS_UA = {
    "dog": "собака",
    "cat": "кіт",
    "kitten": "кошеня",
    "puppy": "цуценя",
    "horse": "кінь",
    "car": "авто",
    "truck": "вантажівка",
    "bus": "автобус",
    "bicycle": "велосипед",
    "motorcycle": "мотоцикл",
    "plane": "літак",
    "airplane": "літак",
    "bird": "птах",
    "ship": "корабель",
    "boat": "човен",
    "snake": "змія",
    "spider": "павук",
    "fish": "риба",
    "bear": "ведмідь",
    "fox": "лисиця",
    "wolf": "вовк",
    "tiger": "тигр",
    "lion": "лев",
    "elephant": "слон",
    "cow": "корова",
    "sheep": "вівця",
    "monkey": "мавпа",
    "keyboard": "клавіатура",
    "laptop": "ноутбук",
    "phone": "телефон",
    "camera": "камера",
    "bottle": "пляшка",
    "cup": "чашка",
    "chair": "стілець",
    "table": "стіл",
    "pizza": "піца",
    "burger": "бургер",
    "sandwich": "сендвіч",
}

def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def _to_ua_label(label_en: str) -> str:
    # ImageNet labels зазвичай: "golden_retriever", "sports_car"
    nice = label_en.replace("_", " ")
    words = nice.split()
    ua_words = [_WORDS_UA.get(w.lower(), w) for w in words]
    if ua_words != words:
        return " ".join(ua_words)
    return nice

def get_model():
    # Ваги будуть автоматично завантажені Keras при першому запуску
    return mobilenet_v2.MobileNetV2(weights="imagenet")

def preprocess(img: Image.Image) -> np.ndarray:
    img = _ensure_rgb(img)
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = mobilenet_v2.preprocess_input(arr)
    return arr

def predict_topk(image: Image.Image, model, top_k: int = 5) -> List[Dict]:
    x = preprocess(image)
    preds = model.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=top_k)[0]  # (class_id, label, score)
    out: List[Dict] = []
    for _, label, score in decoded:
        out.append(
            {
                "label_en": label,
                "label_ua": _to_ua_label(label),
                "score": float(score),
            }
        )
    return out
