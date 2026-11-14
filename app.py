

import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Student Placement AI",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F3F6FC;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #15345B;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #64748B;
    font-size: 17px;
}

.result-card {
    background: linear-gradient(120deg, #15345B, #2876B9);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.stButton > button {
    background: #1769AA;
    color: white;
    border-radius: 10px;
    height: 48px;
    font-weight: bold;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("Trained_Model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">Student Placement Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Student Placement Analysis System</div>',
    unsafe_allow_html=True
)

st.write("")
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("About Project")
st.sidebar.info(
    "This application uses a trained "
    "Logistic Regression model to predict "
    "student placement outcomes."
)

st.sidebar.write("**Algorithm:** Logistic Regression")
st.sidebar.write("**Technology:** Python + Streamlit")
st.sidebar.write("**Model:** Trained_Model.pkl")

# ---------------- DASHBOARD ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("ML Algorithm", "Logistic Regression")

with col2:
    st.metric("Model Status", "Loaded")

with col3:
    st.metric("Prediction Type", "Classification")

st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("Student Information")
st.write("Enter the student's information below.")

features = list(model.feature_names_in_)

input_values = {}

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        with (col1 if i % 2 == 0 else col2):

            input_values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=0.0,
                step=0.1,
                format="%.2f"
            )

    submitted = st.form_submit_button(
        "Predict Student Placement",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if submitted:

    input_data = pd.DataFrame(
        [input_values],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Prediction Results")

    st.markdown(
        f"""
        <div class="result-card">
            <h2>Prediction Completed</h2>
            <h1>{prediction}</h1>
            <p>Predicted Class</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]

        st.write("")
        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({
            "Class": [str(c) for c in model.classes_],
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Class")
        )

        for label, probability in zip(
            model.classes_, probabilities
        ):
            st.write(
                f"Class {label}: {probability * 100:.2f}%"
            )

# ---------------- FOOTER ----------------
st.divider()

st.markdown(
    "<center>Developed with Python, Scikit-learn "
    "and Streamlit | Student Placement AI Project</center>",
    unsafe_allow_html=True
)
