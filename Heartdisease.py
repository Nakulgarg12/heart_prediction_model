import streamlit as st
import pandas as pd 
import numpy as np 
import pickle
import random 
st.header('Heart Disease Prediction Using Machine Learning ')
data = '''Heart Disease Prediction using Machine Learning Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:

**Logistic Regression**

**Naive Bayes**

**Support Vector Machine (Linear)**

**K-Nearest Neighbors**

**Decision Tree**

**Random Forest**

**XGBoost**

**Artificial Neural Network (1 Hidden Layer, Keras)**

'''
st.markdown(data)
st.image('https://i.pinimg.com/originals/6d/e1/7f/6de17f9493638040838c12f4c947365b.gif')
with open('heart_disease_pred.pkl','rb') as f:
    chatgpt = pickle.load(f)
#load data 
url = url = '''https://github.com/Nakulgarg12/heart_prediction_model/blob/main/heart.csv?raw=true'''
df = pd.read_csv(url)
st.sidebar.header('Select features to predict Heart Disease')
st.sidebar.image('https://i.pinimg.com/originals/26/1b/48/261b48df90def9d5a619bdd54fc23f1d.gif')
all_values = []

for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

final_value =[all_values]
ans = chatgpt.predict(final_value)[0]
import time

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease')

place = st.empty()
place.image('https://i.pinimg.com/originals/4d/e5/67/4de5677c2ab0f5b2f48b0933c55a2da9.gif',width = 200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'No Heart Disease Detected'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'Heart Disease Found'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)
