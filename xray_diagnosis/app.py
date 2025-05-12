import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Английские и русские названия диагнозов
CLASS_NAMES_EN = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

CLASS_NAMES_RU = [
    'Ателектаз', 'Кардиомегалия', 'Консолидация', 'Отёк', 'Экссудат',
    'Эмфизема', 'Фиброз', 'Грыжа', 'Инфильтрат', 'Опухоль',
    'Узел', 'Утолщение плевры', 'Пневмония', 'Пневмоторакс', 'Без патологии'
]

IMG_SIZE = (224, 224)
MODEL_PATH = "xray_model.keras"

st.set_page_config(page_title="Классификация заболеваний по рентгену", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# 🌐 Переключатель языка
lang = st.selectbox("🌐 Выберите язык / Select language", ["Русский", "English"])
if lang == "Русский":
    class_names = CLASS_NAMES_RU
    upload_label = "📤 Загрузите изображение"
    analyzing_text = "### ⏳ Анализируем изображение..."
    result_title = "## 🧾 Результаты классификации"
    image_caption = "🖼 Загруженное изображение"
    page_title = "Классификация заболеваний по рентгену"
    instructions = "Загрузите изображение грудной клетки, и модель покажет вероятность каждого заболевания."
else:
    class_names = CLASS_NAMES_EN
    upload_label = "📤 Upload an image"
    analyzing_text = "### ⏳ Analyzing the image..."
    result_title = "## 🧾 Classification Results"
    image_caption = "🖼 Uploaded Image"
    page_title = "X-ray Disease Classification"
    instructions = "Upload a chest X-ray image, and the model will show the probability of each disease."

# CSS для прогрессбаров
st.markdown("""
    <style>
        .label {
            font-weight: 600;
            font-size: 16px;
        }
        .bar-container {
            background-color: #e0e0e0;
            border-radius: 5px;
            height: 20px;
            margin-top: 5px;
            margin-bottom: 15px;
        }
        .bar {
            height: 100%;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 " + page_title)
st.markdown(instructions)

uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=image_caption, use_column_width=True)
    st.markdown(analyzing_text)

    model = load_model()
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)[0]

    st.markdown(result_title)

    # Сортировка по вероятности убыванию
    sorted_indices = np.argsort(prediction)[::-1]
    for idx in sorted_indices:
        name = class_names[idx]
        percent = prediction[idx] * 100
        color = "#d9534f" if percent > 60 else "#f0ad4e" if percent > 30 else "#5bc0de"
        st.markdown(
            f"""
            <div class="label">{name}: <span style='color:{color}'>{percent:.2f}%</span></div>
            <div class="bar-container">
                <div class="bar" style="width:{percent}%; background-color:{color}"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
