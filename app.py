import streamlit as st
from groq import Groq
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# --- 1. BACKEND: Load and Train Model ---
@st.cache_resource # Keeps the model in memory so it's fast
def load_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X, data.feature_names

model, X_train, feature_names = load_model()

# --- 2. FRONTEND: Sidebar Inputs ---
st.title("🩺 Healthcare XAI Dashboard")
st.write("Input patient metrics to get an explainable diagnosis.")

st.sidebar.header("Patient Measurements")
# Create sliders for the top 5 most important features
input_dict = {}
for feat in feature_names[:5]: # Simplified to first 5 for the demo
    input_dict[feat] = st.sidebar.slider(f"Value for {feat}", 
                                         float(X_train[feat].min()), 
                                         float(X_train[feat].max()), 
                                         float(X_train[feat].mean()))

# --- 3. MIDDLEWARE: Prediction & SHAP ---
user_input = pd.DataFrame([input_dict])
# Fill missing features with averages so the model can run
full_input = pd.DataFrame([X_train.mean()], columns=feature_names)
full_input.update(user_input)

if st.button("Predict Diagnosis"):
    prediction = model.predict(full_input)[0]
    prob = model.predict_proba(full_input)[0][1]
    
    # Result Display
    result = "Malignant" if prediction == 1 else "Benign"
    st.subheader(f"Diagnosis: **{result}**")
    st.write(f"Confidence Level: {prob:.2%}")

    # SHAP Explanation
    st.divider()
    st.subheader("Why did the AI choose this?")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(full_input)
    
    # Render the Waterfall Plot
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0][:, 1], show=False)
    st.pyplot(plt.gcf())
    
    # --- CHATBOT SECTION ---'
# 1. Setup GenAI Client (Replace with your actual API key)
client = Groq(api_key="YOUR_GROQ_API_KEY")

st.header("🤖 GenAI Medical Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about the clinical results..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Contextual Prompting (This is the "GenAI magic")
    # We feed the diagnosis and SHAP values into the LLM
    context = f"""
    The AI model diagnosed the patient as {result}.
    The confidence is {prob:.2%}.
    The most important features from SHAP are: {feature_names[:3]}.
    User Question: {prompt}
    Please explain this clearly to a medical professional.
    """

    with st.chat_message("assistant"):
        # 3. Call the Generative AI model
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": context}],
            model="llama3-8b-8192", # Using a fast Llama model
        )
        response = chat_completion.choices[0].message.content
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})