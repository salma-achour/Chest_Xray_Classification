import streamlit as st 
import tensorflow as tf 
import cv2
from PIL import Image, ImageOps
import numpy as np 

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('model.hdf5')
    return model
model = load_model()

st.write("""
# Chest X-Xrays Classification
""")

file = st.file_uploader("Please upload your chest X-ray image", type=['jpg', 'png'])

def import_and_predict(image_data, model):

    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    col_scale = image.convert('RGB')
    img = np.asarray(col_scale)
    img_reshape = img.reshape(1,224,224,3)
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ["COVID-19", "NORMAL", "VIRAL PNEUMONIA"]
    strings = "This image most likely is " + class_names[np.argmax(prediction)]
    st.success(strings)