import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Cancer Classifier", page_icon="🔬", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Erreur : {e}")

st.title("🔬 Classificateur de Tumeurs Mammaires")
st.markdown("""
**Modèle :** Régression Logistique | **Accuracy** 98.25% | **ROC-AUC** 0.9954  
**Dataset :** Breast Cancer Wisconsin  
Entrez les caractéristiques cellulaires pour obtenir une prédiction.
""")
st.divider()

if not model_loaded:
    st.stop()

FEATURE_NAMES = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    'radius error','texture error','perimeter error','area error','smoothness error',
    'compactness error','concavity error','concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry','worst fractal dimension'
]

EXAMPLE_BENIGN = [
    13.37,16.32,86.20,555.0,0.09276,0.06645,0.04974,0.03080,0.1765,0.06156,
    0.2521,1.197,1.717,22.07,0.006881,0.01650,0.02080,0.009806,0.01646,0.002974,
    15.11,21.37,97.65,711.4,0.1297,0.1862,0.2400,0.1194,0.2764,0.08158
]
EXAMPLE_MALIGNANT = [
    20.29,14.34,135.1,1297.0,0.1003,0.1328,0.1980,0.1043,0.1809,0.05883,
    0.7572,0.7813,5.438,94.44,0.01149,0.02461,0.05688,0.01885,0.01756,0.005115,
    22.54,16.67,152.2,1575.0,0.1374,0.2050,0.4000,0.1625,0.2364,0.07678
]

with st.sidebar:
    st.header("⚙️ Options")
    mode = st.radio("Mode", ["Exemple prédéfini", "Saisie manuelle"])
    if mode == "Exemple prédéfini":
        example = st.selectbox("Choisir", ["Patient bénin","Patient malin"])

groups = {
    "🔵 Moyennes (mean)": FEATURE_NAMES[:10],
    "🟡 Erreurs standard (SE)": FEATURE_NAMES[10:20],
    "🔴 Pires valeurs (worst)": FEATURE_NAMES[20:30],
}

if mode == "Exemple prédéfini":
    defaults = EXAMPLE_BENIGN if "bénin" in example.lower() else EXAMPLE_MALIGNANT
else:
    defaults = [0.0]*30

input_values = []
for group_name, group_features in groups.items():
    with st.expander(group_name, expanded=(group_name=="🔵 Moyennes (mean)")):
        cols = st.columns(2)
        for j, feat in enumerate(group_features):
            idx = FEATURE_NAMES.index(feat)
            val = cols[j%2].number_input(feat, value=float(defaults[idx]), format="%.5f", key=f"f_{idx}")
            input_values.append((idx, val))

input_ordered = [v for _,v in sorted(input_values)]
X_input = np.array(input_ordered).reshape(1,-1)
X_scaled = scaler.transform(X_input)

st.divider()
if st.button("🔍 Prédire", type="primary"):
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    col1, col2, col3 = st.columns(3)
    if prediction == 1:
        col1.success("## ✅ BÉNIN")
    else:
        col1.error("## ⚠️ MALIN")
    col2.metric("Probabilité Bénin", f"{proba[1]:.2%}")
    col3.metric("Probabilité Malin", f"{proba[0]:.2%}")
    st.progress(float(max(proba)), text=f"{max(proba):.1%} de confiance")
    st.info("🏥 **Avertissement** : Ceci est un outil académique, pas un diagnostic médical.")

st.divider()
st.caption("Projet ML — Breast Cancer Wisconsin | Régression Logistique")
