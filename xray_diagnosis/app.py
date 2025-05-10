import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ====== Константы ======
MODEL_PATH = "PythonProject/xray_model.keras"  # путь до модели
IMG_SIZE = (224, 224)  # размер изображения (замени, если у тебя другой)
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]  # замени на свои классы

# ====== Загрузка модели ======
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# ====== Интерфейс ======
st.title("🩻 Классификация заболеваний по рентгену")
st.write("Загрузите изображение грудной клетки для анализа моделью.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    # ====== Предобработка ======
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ====== Предсказание ======
    predictions = model.predict(img_array)[0]

    # ====== Результаты ======
    st.subheader("Результаты:")
    for class_name, prob in zip(CLASS_NAMES, predictions):
        st.write(f"**{class_name}**: {prob:.2%}")
