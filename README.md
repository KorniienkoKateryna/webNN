# Streamlit Image Classifier (Keras Applications)

Web‑додаток на Streamlit для розпізнавання зображень з кнопкою «Розпізнати».

## Особливості
- **Keras Applications**: MobileNetV2 (ImageNet)
- Інтерфейс українською, Top‑K результати
- Працює у Streamlit Cloud на **Python 3.13** завдяки **Keras 3 + Torch backend** (без TensorFlow)

## Локальний запуск
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
