import streamlit as st
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
from PIL import Image

# Load the TFLite model
interpreter = Interpreter(model_path="mnist_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("MNIST Digit Classifier")
uploaded_file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    image_array = np.array(image).reshape(1, 28, 28, 1).astype('float32') / 255

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)
    # Run inference
    interpreter.invoke()
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_digit = np.argmax(output_data)

    st.write(f"Predicted Digit: {predicted_digit}")
