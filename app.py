import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the trained model (you'll need to save and load the model properly)
model = GradientBoostingClassifier()  # Replace with your trained model

st.title('Diabetes Prediction App')

# Create input fields for each feature
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=300, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=30)

# Create a dataframe with the input values
input_data = pd.DataFrame({
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree],
    'Age': [age]
})

input_df = pd.DataFrame(input_data)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

x_trans = scaler.transform(input_df)

with open('best_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
# Make prediction
if st.button('Predict'):
    prediction = model.predict(x_trans)
    if prediction[0] == 0:
        st.write('The model predicts: No diabetes')
    else:
        st.write('The model predicts: Diabetes')