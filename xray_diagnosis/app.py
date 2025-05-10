import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# === Настройки страницы (должно быть ПЕРВОЙ строчкой) ===
st.set_page_config(page_title="Диагностика по рентгену", layout="centered")

# === Константы ===
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
    'Unknown'  # добавлен 15-й класс
]
MODEL_PATH = "xray_model.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# === Загрузка модели с кэшированием ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# === Предобработка изображения ===
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image_array, axis=0)

# === Grad-CAM ===
def generate_gradcam(model, img_array, class_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=-3).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam.numpy(), IMG_SIZE)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

# === Гистограмма вероятностей ===
def plot_probabilities(preds):
    fig, ax = plt.subplots(figsize=(10, 6))
    n_classes = min(len(CLASS_NAMES), len(preds))
    colors = ["red" if preds[i] >= THRESHOLD else "gray" for i in range(n_classes)]
    ax.barh(CLASS_NAMES[:n_classes], preds[:n_classes], color=colors)
    ax.set_xlim([0, 1])
    ax.invert_yaxis()
    ax.set_xlabel("Вероятность")
    ax.set_title("Предсказания модели")
    st.pyplot(fig)

# === Интерфейс ===
st.title("🩻 Классификация заболеваний по рентгену")
st.write("Загрузите изображение грудной клетки для анализа моделью.")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Исходное изображение", use_column_width=True)
    img_array = preprocess_image(image)

    preds = model.predict(img_array)[0]
    n_classes = len(CLASS_NAMES)

    if len(preds) != n_classes:
        st.error(f"Ошибка: модель вернула {len(preds)} значений, а классов {n_classes}.")
    else:
        plot_probabilities(preds)

        high_preds = [(cls, float(prob)) for cls, prob in zip(CLASS_NAMES, preds) if prob >= THRESHOLD]

        if high_preds:
            st.markdown("### 🩺 Обнаружено:")
            for cls, prob in high_preds:
                st.write(f"**{cls}**: {prob*100:.2f}%")
        else:
            st.info("Серьёзные патологии не выявлены (все вероятности < 50%).")

        # Grad-CAM для самого вероятного класса
        top_class = int(np.argmax(preds))
        cam = generate_gradcam(model, img_array, top_class)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        orig = np.array(image.resize(IMG_SIZE))
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

        st.markdown("### 🌡 Grad-CAM визуализация")
        st.image(overlay, caption=f"Область внимания модели: {CLASS_NAMES[top_class]}", use_column_width=True)
