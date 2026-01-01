# Streamlit Image Classifier (Keras Applications)

Це web‑додаток на **Streamlit** для розпізнавання зображень.

## Модель
- **MobileNetV2** з `keras.applications`
- Попередньо навчена на **ImageNet (~1000 класів)**
- Приклади класів: тварини, транспорт, предмети побуту, їжа, інструменти тощо.

## Як працює
1) Користувач завантажує зображення / робить фото
2) Зображення масштабується до 224×224 та нормалізується
3) Модель повертає **Top‑K** класів з ймовірностями
4) Результати відображаються українською (частково) + англ. оригінальні назви

## Важливо (для Streamlit Cloud)
Streamlit Cloud зараз використовує Python 3.13.  
Щоб уникнути проблем із TensorFlow, використано **Keras 3 + Torch backend** (через `KERAS_BACKEND=torch`).

## Локальний запуск
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
