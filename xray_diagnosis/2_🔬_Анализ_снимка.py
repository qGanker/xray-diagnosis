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

# Интерфейс
st.set_page_config(page_title="Классификация заболеваний по рентгену", layout="centered")
st.title("💀 Классификация заболеваний по рентгену")
st.write("Загрузите изображение грудной клетки для анализа моделью.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    st.write("🔍 Анализируем...")

    model = load_model()
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)[0]

    st.subheader("Результаты:")
    for name, prob in zip(CLASS_NAMES, prediction):
        st.write(f"**{name}**: {prob * 100:.2f}%")