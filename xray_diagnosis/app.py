import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Классы заболеваний
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

IMG_SIZE = (224, 224)
MODEL_PATH = "xray_model.keras"

# Настройка страницы
st.set_page_config(page_title="Классификация заболеваний по рентгену", layout="centered")

# Кешируем загрузку модели
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

# Предобработка изображения
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Стилизация CSS
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stFileUploader {
            margin-top: 1em;
        }
        .prob-card {
            padding: 10px;
            border-radius: 10px;
            background-color: #ffffff;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .high { color: #d9534f; font-weight: bold; }
        .medium { color: #f0ad4e; font-weight: bold; }
        .low { color: #5bc0de; }
    </style>
""", unsafe_allow_html=True)

# Заголовок
st.title("🩺 Классификация заболеваний по рентгену")
st.markdown("Загрузите изображение грудной клетки, и модель покажет вероятность каждого заболевания.")

# Загрузка изображения
uploaded_file = st.file_uploader("📤 Загрузите изображение (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Загруженное изображение", use_column_width=True)
    st.markdown("### ⏳ Анализируем изображение...")

    model = load_model()
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)[0]

    # Показать результаты
    st.markdown("## 🧾 Результаты классификации")
    for name, prob in zip(CLASS_NAMES, prediction):
        percent = prob * 100
        if percent > 60:
            style = "high"
        elif percent > 30:
            style = "medium"
        else:
            style = "low"
        st.markdown(
            f'<div class="prob-card"><span class="{style}">{name}</span>: {percent:.2f}%</div>',
            unsafe_allow_html=True
        )
