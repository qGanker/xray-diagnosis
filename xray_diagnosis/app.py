import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# === Константы ===
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'No Finding'
]
IMG_SIZE = (224, 224)
MODEL_PATH = "xray_model.keras"
THRESHOLD = 0.5

# === Настройка страницы ===
st.set_page_config(page_title="Классификация заболеваний по рентгену", layout="centered")
st.title("🩻 Классификация заболеваний по рентгену")
st.write("Загрузите изображение грудной клетки для анализа моделью.")

# === Кэшируем модель ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# === Предобработка изображения ===
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# === Grad-CAM ===
def generate_gradcam(model, img_array, class_index):
    try:
        conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                  tf.keras.layers.SeparableConv2D,
                                  tf.keras.layers.DepthwiseConv2D)):
                conv_layer = layer.name
                break
        if conv_layer is None:
            st.warning("❗ Не найден сверточный слой для Grad-CAM.")
            return None

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(conv_layer).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            st.warning("❗ Не удалось вычислить градиенты (grads = None).")
            return None

        grads = grads[0]
        conv_outputs = conv_outputs[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.zeros(conv_outputs.shape[:2], dtype=tf.float32)

        for i in range(weights.shape[0]):
            cam += weights[i] * conv_outputs[:, :, i]

        cam = tf.nn.relu(cam)
        cam = tf.image.resize(cam[..., tf.newaxis], IMG_SIZE)
        cam = tf.squeeze(cam)
        cam = (cam - tf.reduce_min(cam)) / (tf.reduce_max(cam) - tf.reduce_min(cam) + 1e-8)
        return cam.numpy()
    except Exception as e:
        st.warning(f"Grad-CAM ошибка: {e}")
        return None

# === Загрузка изображения ===
uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Исходное изображение", use_column_width=True)

    preprocessed = preprocess_image(image)
    preds = model.predict(preprocessed)[0]

    if len(preds) != len(CLASS_NAMES):
        st.error(f"Ошибка: модель вернула {len(preds)} значений, а классов {len(CLASS_NAMES)}.")
    else:
        st.subheader("Результаты:")
        for name, prob in zip(CLASS_NAMES, preds):
            st.write(f"**{name}**: {prob * 100:.2f}%")

        # === Grad-CAM по топ-классу с вероятностью выше порога ===
        filtered = [(i, p) for i, p in enumerate(preds) if p > THRESHOLD]
        top_index = filtered[0][0] if filtered else int(np.argmax(preds))

        cam = generate_gradcam(model, preprocessed, top_index)
        if cam is not None:
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            orig = np.array(image.resize(IMG_SIZE))
            overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
            st.subheader("🌡 Область внимания модели")
            st.image(overlay, caption=f"Grad-CAM для класса: {CLASS_NAMES[top_index]}", use_column_width=True)
        else:
            st.warning("Не удалось построить Grad-CAM визуализацию для этого изображения.")
