import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('mnist_model.h5')

# Streamlit UI
st.title("MNIST Digit Classifier")
uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255
    prediction = model.predict(image_array)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")
