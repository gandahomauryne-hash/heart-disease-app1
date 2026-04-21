import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀")
st.title("🫀 Prédiction de Maladie Cardiaque")
st.write("Remplis les informations ci-dessous pour obtenir une prédiction.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
    cp = st.selectbox("Type de douleur thoracique (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Pression arterielle (mm Hg)", 90, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Glycemie > 120 mg/dl", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
    restecg = st.selectbox("ECG au repos", [0, 1, 2])

with col2:
    thalach = st.slider("Frequence Cardiaque Maximale", 70, 200, 150)
    exang = st.selectbox("Angine de poitrine induite par l'exercice", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui")
    oldpeak = st.slider("Depression du segment ST", 0.0, 6.0, 1.0)
    slope = st.selectbox("Pente du segment ST", [0, 1, 2])
    ca = st.selectbox("Nombre de vaisseaux majeurs (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalessemie", [1, 2, 3], format_func=lambda x: "Normal" if x == 1 else ("Defaut fixe" if x == 2 else "Defaut reversible"))

# Prédiction
if st.button("Prédire"):
    input_data = np.array([
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)

    st.subheader("Résultat de la prédiction")
    if prediction[0] == 1:
        st.error("⚠️ Forte probabilité de maladie cardiaque.")
    else:
        st.success("✅ Faible probabilité de maladie cardiaque.")

    st.write(f"Probabilité de ne pas avoir de maladie cardiaque : {prediction_proba[0][0]:.2%}")
    st.write(f"Probabilité d'avoir une maladie cardiaque : {prediction_proba[0][1]:.2%}")


