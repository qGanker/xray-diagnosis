import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Классы и переводы
CLASS_NAMES_EN = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]

CLASS_NAMES_RU = [
    'Ателектаз', 'Кардиомегалия', 'Консолидация', 'Отёк', 'Эффузия',
    'Эмфизема', 'Фиброз', 'Грыжа', 'Инфильтрация', 'Опухоль',
    'Узел', 'Плевральное утолщение', 'Пневмония', 'Пневмоторакс', 'Без патологии'
]

EXPLANATIONS_RU = {
    'Ателектаз': "Участок лёгкого спавшийся или без воздуха, часто из-за обструкции бронха.",
    'Кардиомегалия': "Увеличенное сердце, может указывать на сердечную недостаточность.",
    'Консолидация': "Уплотнение лёгочной ткани из-за воспаления или жидкости.",
    'Отёк': "Жидкость в лёгочной ткани, часто при сердечной недостаточности.",
    'Эффузия': "Скопление жидкости в плевральной полости.",
    'Эмфизема': "Разрушение альвеол, связанное с ХОБЛ и курением.",
    'Фиброз': "Утолщение и рубцевание лёгочной ткани.",
    'Грыжа': "Выход органов через грудную стенку или диафрагму.",
    'Инфильтрация': "Неспецифические изменения, связанные с инфекцией или опухолью.",
    'Опухоль': "Объёмное образование, требующее дополнительной диагностики.",
    'Узел': "Маленькое плотное образование, может быть доброкачественным или злокачественным.",
    'Плевральное утолщение': "Утолщение оболочек лёгких, может быть от хронического воспаления.",
    'Пневмония': "Инфекция лёгочной ткани.",
    'Пневмоторакс': "Воздух в плевральной полости, вызывает спадение лёгкого.",
    'Без патологии': "Признаков заболеваний не обнаружено."
}

EXPLANATIONS_EN = {
    'Atelectasis': "Collapsed or airless portion of the lung, often due to bronchial obstruction.",
    'Cardiomegaly': "Enlarged heart, may indicate heart failure.",
    'Consolidation': "Lung tissue filled with liquid instead of air due to inflammation.",
    'Edema': "Fluid buildup in lungs, commonly caused by heart failure.",
    'Effusion': "Fluid in the pleural space around the lungs.",
    'Emphysema': "Lung damage from air sac destruction, linked to smoking or COPD.",
    'Fibrosis': "Scarring and thickening of lung tissue.",
    'Hernia': "Protrusion of an organ through the chest wall or diaphragm.",
    'Infiltration': "Nonspecific lung shadowing, may relate to infection or cancer.",
    'Mass': "A larger abnormal growth in the lungs.",
    'Nodule': "A small round spot in the lung, benign or malignant.",
    'Pleural_Thickening': "Thickened lung lining, possibly from chronic inflammation.",
    'Pneumonia': "Infection causing inflammation of the lungs.",
    'Pneumothorax': "Air between lung and chest wall, can collapse the lung.",
    'No Finding': "No abnormalities detected."
}

IMG_SIZE = (224, 224)
MODEL_PATH = "xray_model.keras"

# Конфигурация страницы
st.set_page_config(page_title="Классификация заболеваний по рентгену", layout="centered")

# Кэшируем модель
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# Стилизация
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
            margin-bottom: 5px;
        }
        .bar {
            height: 100%;
            border-radius: 5px;
        }
        .explanation {
            font-size: 14px;
            color: #ccc;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Заголовки
st.title("🩺 Классификация заболеваний по рентгену")

# Переключатель языка
lang = st.radio("🌐 Выберите язык / Select language", ["Русский", "English"], horizontal=True)

st.markdown(
    "Загрузите рентгеновский снимок грудной клетки, чтобы получить вероятности заболеваний."
    if lang == "Русский"
    else "Upload a chest X-ray to get disease probabilities."
)

uploaded_file = st.file_uploader("📤 Загрузите изображение / Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Загруженное изображение" if lang == "Русский" else "🖼 Uploaded Image", use_column_width=True)

    st.markdown("### ⏳ Анализируем изображение..." if lang == "Русский" else "### ⏳ Analyzing image...")

    model = load_model()
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)[0]
    sorted_indices = np.argsort(-prediction)

    st.markdown("## 🧾 Результаты классификации" if lang == "Русский" else "## 🧾 Classification Results")

    class_names = CLASS_NAMES_RU if lang == "Русский" else CLASS_NAMES_EN
    explanations = EXPLANATIONS_RU if lang == "Русский" else EXPLANATIONS_EN

    for idx in sorted_indices:
        name = class_names[idx]
        percent = prediction[idx] * 100
        color = "#d9534f" if percent > 60 else "#f0ad4e" if percent > 30 else "#5bc0de"
        explanation = explanations.get(name, "🔍 Нет объяснения." if lang == "Русский" else "🔍 No explanation available.")

       st.markdown(
    f"""
    <div class="label">{name}: <span style='color:{color}'>{percent:.2f}%</span></div>
    <div class="bar-container">
        <div class="bar" style="width:{percent}%; background-color:{color};"></div>
    </div>
    <div class="explanation">{explanation}</div>
    """,
    unsafe_allow_html=True
)

