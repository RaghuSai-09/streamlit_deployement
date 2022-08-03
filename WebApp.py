# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pickle
import pandas as pd
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

import plotly.express as px


#stemming
ps = PorterStemmer() 

#loading dataset 

df = pd.read_csv("data.csv")
dataset_df = pd.read_csv("Dataset.csv")



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


 
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model1 = pickle.load(open('BNB_model.pkl','rb'))
model2 = pickle.load(open('MNB_model.pkl','rb'))

# selecting model from selectbox
st.sidebar.title('MODEL')
select_model = st.sidebar.selectbox("Select Model", ("BernoulliNB","MultinomialNB"))

if select_model=="BernoulliNB":
    model =  model1
    
elif select_model=="MultinomialNB":
    model = model2
    
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam Email/SMS")
    else:
        st.header("Not Spam/Ham")
        


st.write('')
st.header('Dataset')
st.write(dataset_df)
st.write('')
st.header('Result')
st.write(df)

st.title("Accuracy Comparison Plot")
fig = px.bar(df,x='Model',y='Accuracy',color='Model',range_y=(0.5,1.2))
st.write(fig)

st.title("Precision Comparison Plot")
fig1 = px.bar(df,x='Model',y='Precision',color='Model',range_y=(0.3,1.2))
st.write(fig1)

st.title('Accuracy Vs Precision')
fig2 = px.line(df,x='Model',y=['Accuracy', 'Precision'],labels={'x':'Model','y':'Value'},range_y=(0.3,1.2))
st.write(fig2)
