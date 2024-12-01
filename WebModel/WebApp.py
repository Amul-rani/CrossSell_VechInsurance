import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Cross Sell Prediction App")

df = pd.read_csv('train.csv')

gender = st.selectbox("Gender",pd.unique(df['Gender']))
age = st.number_input("Age")
driving_license   = st.number_input("Driving_License")
region_code = st.number_input("Region_Code")
previously_insured  = st.number_input("Previously_Insured")
vehicle_age  = st.selectbox("Vehicle_Age",pd.unique(df['Vehicle_Age']))
vehicle_damage= st.selectbox("Vehicle_Damage",pd.unique(df['Vehicle_Damage']))
annual_premium = st.number_input("Annual_Premium")
policy_sales_channel= st.number_input("Policy_Sales_Channel")
vintage = st.number_input("Vintage")




inputs = {
'Gender' :gender,
'Age' :age,
'Driving_License' :driving_license,
'Region_Code' :region_code,
'Previously_Insured' :previously_insured,
'Vehicle_Age' :vehicle_age,
'Vehicle_Damage':vehicle_damage,
'Annual_Premium': annual_premium,
'Policy_Sales_Channel' :policy_sales_channel,
'Vintage' :vintage
}

# load the model from pikle file
model =joblib.load('jobchg_pipeline_model.pkl')

if st.button('Predit'):
    X_input = pd.DataFrame(inputs,index=[0])
    prediction = model.predict(X_input)
    st.write("The predicted value is:")
    st.write(prediction)
