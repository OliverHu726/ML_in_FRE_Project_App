import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import BytesIO
import pickle
from PIL import Image
import sklearn

# Add import statements for Preprocessor and columns if necessary

model_url = 'https://raw.github.com/OliverHu726/ML_in_FRE_Project_App/main/model.pkl'
model_response = requests.get(model_url)
model = pickle.load(BytesIO(model_response.content))

# title
st.title("Predictions based on stock sentiment index")
st.write('*By feeding in the sentiment index, the model can predict the future rise and fall of the stock!* :sunglasses:')

Subjectivity = st.number_input('Subjectivity 0~1', 0.0)
Polarity = st.number_input('Polarity -0.5~0.5', -0.4)
compound = st.number_input('compound -1~1', -0.997)
neg = st.number_input('neg 0~1', 0.0)
pos = st.number_input('pos 0~1', 0.0)
neu = st.number_input('neu 0~1', 0.532)

# Make sure to define 'columns' with the correct order of features used during training
columns = ['Subjectivity', 'Polarity', 'compound', 'neg', 'pos', 'neu']

def predict():
    row = np.array([Subjectivity, Polarity, compound, neg, pos, neu])
    X = pd.DataFrame([row], columns=columns)
    prediction = model.predict(X)[0]
    if prediction == 1:
        st.success('Stocks have a higher probability of going up :thumbsup:')
    else:
        st.error('Stocks have a higher probability of going down :thumbsdown:')

# Fix the on_click parameter
st.button('Predict', on_click=predict)

# Load results of Visualization
wc_url = 'https://raw.github.com/OliverHu726/ML_in_FRE_Project_App/main/graphs/wordcloud.png'
wc_response = requests.get(wc_url)
wc = Image.open(BytesIO(wc_response.content))

vi_url = 'https://raw.github.com/OliverHu726/ML_in_FRE_Project_App/main/graphs/VariableImportance.png'
vi_response = requests.get(vi_url)
vi = Image.open(BytesIO(vi_response.content))

st.image(wc, caption='Word Cloud', use_column_width=True)
st.image(vi, caption='Variable Importance', use_column_width=True)



