import streamlit as st
import pickle
import numpy as np
import time

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #FFF5F5 0%, #FFE3E3 100%);
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        color: #C0392B;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .sub-header {
        color: #555;
        text-align: center;
        margin-bottom: 30px;
        font-size: 1.1rem;
    }

    /* Input Card Styling */
    .input-container {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-top: 5px solid #C0392B;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #D93025 0%, #C0392B 100%);
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        color: white;
    }

    /* Result Cards */
    .result-card-danger {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #C62828;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        animation: fadeIn 1s;
    }
    .result-card-safe {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #2E7D32;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        animation: fadeIn 1s;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open('heart_disease_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please run the training script first.")
        return None

model = load_model()

# -----------------------------------------------------------------------------
# 4. SIDEBAR INFO
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=100)
    st.title("HeartGuard AI")
    st.write("---")
    st.markdown("""
    This intelligent system predicts the likelihood of heart disease using **Logistic Regression**.
    
    **Input Parameters:**
    - **CP:** Chest Pain Type
    - **Thal:** Thalassemia
    - **CA:** Major Vessels (0-4)
    - **Thalach:** Max Heart Rate
    - **Oldpeak:** ST Depression
    """)
    st.write("---")
    st.caption("Disclaimer: For educational purposes only. Consult a doctor for medical advice.")

# -----------------------------------------------------------------------------
# 5. MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🫀 Heart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter patient clinical data below for a risk assessment</div>', unsafe_allow_html=True)

if model:
    # --- Input Form ---
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Patient Vitals")
    
    col1, col2 = st.columns(2)

    with col1:
        # Chest Pain: Mapping user friendly names to numbers (0-3)
        cp_options = {
            0: "Typical Angina (0)",
            1: "Atypical Angina (1)",
            2: "Non-anginal Pain (2)",
            3: "Asymptomatic (3)"
        }
        cp_display = st.selectbox("Chest Pain Type (CP)", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
        
        # Max Heart Rate
        thalach = st.slider("Max Heart Rate (Thalach)", min_value=60, max_value=220, value=150)
        
        # Oldpeak
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ST depression induced by exercise relative to rest")

    with col2:
        # Thalassemia: Mapping numbers (1-3)
        thal_options = {
            1: "Fixed Defect (1)",
            2: "Normal (2)",
            3: "Reversable Defect (3)"
        }
        thal_display = st.selectbox("Thalassemia (Thal)", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])
        
        # Major Vessels (CA)
        ca = st.selectbox("Number of Major Vessels (CA)", options=[0, 1, 2, 3, 4], help="Number of major vessels (0-4) colored by flourosopy")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Prediction Section ---
    if st.button("Analyze Risk Profile"):
        # Progress bar for visual effect
        progress_text = "Processing clinical data..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.5)
        my_bar.empty()

        # Prepare input vector: [cp, thal, ca, thalach, oldpeak]
        input_data = [[cp_display, thal_display, ca, thalach, oldpeak]]
        
        # Prediction
        prediction = model.predict(input_data)
        result = prediction[0]
        
        # Probability (Optional, if supported by model)
        try:
            proba = model.predict_proba(input_data)
            confidence = np.max(proba) * 100
        except:
            confidence = 0

        st.markdown("---")

        if result == 1:
            st.markdown(
                f"""
                <div class="result-card-danger">
                    ⚠️ HIGH RISK DETECTED<br>
                    <span style='font-size:16px; font-weight:normal'>The model predicts a high probability of heart disease.</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            if confidence > 0:
                st.write(f"**Model Confidence:** {confidence:.2f}%")
                
        else:
            st.markdown(
                f"""
                <div class="result-card-safe">
                    ✅ LOW RISK / HEALTHY<br>
                    <span style='font-size:16px; font-weight:normal'>The model predicts no presence of heart disease.</span>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.balloons()
            if confidence > 0:
                st.write(f"**Model Confidence:** {confidence:.2f}%")

else:
    st.info("Awaiting model file...")
