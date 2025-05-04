import streamlit as st
import numpy as np 
import tensorflow as tf # type: ignore

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_sex.pkl','rb') as file:
    label_encoder_sex = pickle.load(file)

with open('onehot_encoder_des.pkl','rb') as file:
    onehot_encoder_des = pickle.load(file)

with open('onehot_encoder_unit.pkl','rb') as file:
    onehot_encoder_unit = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# Streamlit

st.title("Salary Prediction")

# User Inputs

designation = st.selectbox("Designation", onehot_encoder_des.categories_[0])

unit = st.selectbox("Unit", onehot_encoder_unit.categories_[0])

sex  = st.selectbox("Gender", label_encoder_sex.classes_)

age = st.slider("Age", 18, 92)
rating = st.slider("Rating", 0, 5)



past_exp = st.slider('Past Experience',0,25)


input_data = pd.DataFrame({
    'DESIGNATION': [designation],
    'UNIT': [unit],
    'SEX' : [label_encoder_sex.transform([sex])[0]],
    'AGE' : [age],
    'PAST EXP' : [past_exp],
    'RATINGS' : [rating],

})

dev_encoded = onehot_encoder_des.transform([[designation]]).toarray()

dev_encoded_df = pd.DataFrame(dev_encoded, columns=onehot_encoder_des.get_feature_names_out(['DESIGNATION']))

unit_encoded = onehot_encoder_unit.transform([[unit]]).toarray()

unit_encoded_df = pd.DataFrame(unit_encoded, columns=onehot_encoder_unit.get_feature_names_out(['UNIT']))





# Remove DESIGNATION and UNIT from the original DataFrame
input_data.drop(['DESIGNATION', 'UNIT'], axis=1, inplace=True)

# Now combine everything
input_data = pd.concat([input_data.reset_index(drop=True), dev_encoded_df, unit_encoded_df], axis=1)


with open('feature_order.pkl', 'rb') as f:
    feature_order = pickle.load(f)

input_data = input_data[feature_order]

#Scale the input data
input_data_scaled =  scaler.transform(input_data)

prediction = model.predict(input_data_scaled)




if st.button('Predict Salary'):
    prediction = model.predict(input_data_scaled, verbose=0)
    predicted_salary = prediction[0][0]
    st.success(f"Predicted Estimated Salary: ${predicted_salary:,.2f}")
