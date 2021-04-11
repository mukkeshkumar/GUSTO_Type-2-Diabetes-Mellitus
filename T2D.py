import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier

from PIL import Image
image = Image.open('gusto.jpg')
st.image(image,use_column_width=True)

st.write("""
# Type 2 Diabetes Mellitus (T2D) 
This app predicts maternal postpartum T2D outcome early during prenatal care.
""")

st.sidebar.header('Patient Input Features')

# Collects patient input features into dataframe
def patient_input_features():
    m_weight_pw26 = st.sidebar.number_input('Weight (kg)', 0.0,250.0,50.0)
    m_height_pw26 = st.sidebar.number_input('Height (cm)', 0.0,250.0,170.0)
    m_BMI_pw26 = m_weight_pw26/((m_height_pw26*0.01)*(m_height_pw26*0.01))
    m_gdm_who_1999_cat = st.sidebar.selectbox('Diagnosis of Gestational Diabetes Mellitus',('0.0','1.0'))
    data = {'m_BMI_pw26': m_BMI_pw26,
            'm_gdm_who_1999_cat': m_gdm_who_1999_cat}
    features = pd.DataFrame(data, index=[0])
    return features
df = patient_input_features()

df = df[:1] # Selects only the first row (the patient input data)

# Displays the patient input features
st.subheader('Patient Input features :')

st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('mg_t2dyr48_22_auc_cb_2.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)

st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of T2D'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of T2D'] = 'Yes'
st.write(df1)

prediction_proba = load_clf.predict_proba(df)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)
