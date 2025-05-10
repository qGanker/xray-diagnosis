import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Классы заболеваний
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

IMG_SIZE = (224, 224)
MODEL_PATH = "xray_model.keras"

st.set_page_config(page_title="Классификация заболеваний по рентгену", layout="centered")
st.title("💀 Классификация заболеваний по рентгену")
st.write("Загрузите изображение грудной клетки для анализа моделью.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

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

    # Grad-CAM для наиболее вероятного класса
    top_index = int(np.argmax(prediction))
    cam = generate_gradcam(model, preprocessed, top_index)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    original = np.array(image.resize(IMG_SIZE))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    st.subheader("🌡 Grad-CAM визуализация")
    st.image(overlay, caption=f"Область внимания модели: {CLASS_NAMES[top_index]}", use_column_width=True)
