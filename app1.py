import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# --- BACKEND ---
@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    return model, X, data.feature_names

model, X_train, feature_names = train_model()

# --- FRONTEND (Sidebar Inputs) ---
st.set_page_config(page_title="Patient Safety Dashboard", layout="wide")
st.title("🛡️ Patient Safety XAI Monitoring")

with st.sidebar:
    st.header("Patient Vital Signs")
    # Dynamically create inputs for the top medical indicators
    vitals = {}
    for i in range(5):
        name = feature_names[i]
        vitals[name] = st.slider(f"{name}", float(X_train[name].min()), float(X_train[name].max()))

# --- MIDDLE-END (XAI Logic) ---
# Prepare data for prediction
current_patient = pd.DataFrame([X_train.mean()], columns=feature_names)
for k, v in vitals.items():
    current_patient[k] = v

prediction = model.predict(current_patient)[0]
risk_score = model.predict_proba(current_patient)[0][1]

# Display Results
col1, col2 = st.columns(2)
with col1:
    st.subheader("Clinical Prediction")
    status = "⚠️ HIGH RISK" if prediction == 0 else "✅ STABLE"
    st.metric("Patient Status", status)
    st.progress(risk_score)

with col2:
    st.subheader("Explanation of Risk")
    # Calculate SHAP values for the "Middle-End" logic
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(current_patient)
    
    # Plotting
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0][:, 1], show=False)
    st.pyplot(fig)