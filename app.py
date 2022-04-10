import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
import seaborn as sns
import datetime
import streamlit as st

from PIL import Image

# Open files

with open('parametre.yaml') as yaml_data:
    params = yaml.safe_load(yaml_data)
    

# Params

DATA_DIR = pathlib.Path(params['DATA_DIR'])
IMAGE_WIDTH = params['IMAGE_WIDTH']
IMAGE_HEIGHT = params['IMAGE_HEIGHT']
IMAGE_DEPTH = params['IMAGE_DEPTH']
MODELS_DIR = params['MODELS_DIR']
TARGET_NAME = params['TARGET_NAME']

# Function

def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)

def predict_image(path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): path to image to identify
    model (keras.models): Keras model to be used for prediction
    
    Returns
    -------
    Predicted class
    """
    
    images = np.array([np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))])
    prediction_vector = model.predict(images)
    predicted_classes = np.argmax(prediction_vector, axis=1)
    return predicted_classes[0],prediction_vector


def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)


st.title("Aircraft identification")
uploaded_file = st.file_uploader("Upload an aircraft image") #, accept_multiple_files=True)#

if uploaded_file:
    loaded_image = load_image(uploaded_file)
    st.image(loaded_image)
    

choose_data = st.sidebar.selectbox('Choose your Data', ('','Manufacturer', 'Family'),
    index = 0,
    disabled=(uploaded_file is None))

predict_air = st.button("Identify", disabled=(uploaded_file is None))

if choose_data == 'Manufacturer' :
    model = load_model('C:/Users/Camille/Documents/plane-classification/model/manufacturer.h5')
    with open('label_manufacturer.yaml') as file:
        label = yaml.safe_load(file)

if choose_data == 'Family' :
    model = load_model('C:/Users/Camille/Documents/plane-classification/model/family.h5')
    with open('label_family.yaml') as file:
        label = yaml.safe_load(file)
    
if predict_air:

    prediction, prediction_vector = predict_image(uploaded_file, model)
    name_aircraft = label[np.argmax(prediction_vector)]
    st.write(f"The plane on the picture is a : {name_aircraft} with a probability of {round((np.max(prediction_vector))*100,2)}%")

bar_air = st.button('Display the probabilities', disabled=(uploaded_file is None))
if bar_air:
    prediction, prediction_vector = predict_image(uploaded_file, model)
    chart_data = pd.DataFrame(prediction_vector.T, index=list(label.values()))
    st.bar_chart(chart_data)
st.snow()

PATH_MODEL = '../models/' + TARGET_NAME + '.h5'
PATH_CLASSES = '../models/' + TARGET_NAME + '_classes.txt'