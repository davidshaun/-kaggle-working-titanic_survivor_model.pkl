from numpy.ma.core import default_fill_value

import streamlit as st
import pandas as pd
from fastai.tabular.all import

st.title("Titanic Survivorship Predictor")
st.caption("Built With Streamlit")

model = load_learner("titanic_survivor_model.pkl")

def predict_survivor():
    _, pred_idx, probs = model.predict(df.iloc[0])
    confidence = probs[pred_idx]
    return pred_idx, confidence

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Passenger Sex", ["male", "female"])
title = st.selectbox("Passenger Title", model.dls.classes["Title"])
deck = st.slectbox("Cabin Deck", model.dls.classes["Deck"])
embarked = st.selctbox("Embarked", model.dls.classes["Embarked"])

age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare", min_value=0.0, value=30)

SibSp = st.number_input("Siblings / Spouses Aboard", min_value=0.0, value=0.0)
parch = st.number_input("Parents / Children Aboard", min_value=0.0, value=0.0)

age_missing = st.checkbox("Age Unknown")
age_na = 0
if age_missing == True:
    age_na = 1
else:
    age_na = 0

input_data = pd.DataFrame([{
    "Pclass":pclass,
    "Sex":sex,
    "Title":title,
    "Age":age,
    "Age_na":age_na,
    "Fare":fare,
    "SibSp":SibSp,
    "Parch":parch,
    "Embarked":embarked,
}])

if st.button("Predict Survival"):
    pred_idx, confidence = predict_survivor(input_data)
    if pred_idx == 1:
        st.success(f"Survived (confidence{confidence:.2f})")
    else:
        st.error(f"Did Not Survive (confidence{confidence:.2f})")