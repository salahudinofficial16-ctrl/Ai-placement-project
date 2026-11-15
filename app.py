
import sqlite3
import json
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
# import plotly.express as px

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
/* PROFESSIONAL COLORS */

h1, h2, h3 {
    color: #173B67 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF, #DCEBFA);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #B8D4F0;
    box-shadow: 0 4px 12px rgba(20, 60, 100, 0.10);
}

[data-testid="stMetricValue"] {
    color: #087E8B;
    font-weight: bold;
}

.stButton > button {
    background: linear-gradient(90deg, #1769AA, #087E8B);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}

.stNumberInput input {
    border: 2px solid #B8D4F0;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("Trained_Model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()


# ---------------- SQLITE DATABASE ----------------

DB_NAME = "students.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_time TEXT,
            student_data TEXT,
            predicted_class TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


def save_prediction(record):

    conn = sqlite3.connect(DB_NAME)

    conn.execute("""
        INSERT INTO predictions
        (date_time, student_data, predicted_class, confidence)
        VALUES (?, ?, ?, ?)
    """, (
        record["Date & Time"],
        json.dumps({
            key: value for key, value in record.items()
            if key not in ["Date & Time", "Predicted Class", "Confidence (%)"]
        }),
        record["Predicted Class"],
        record["Confidence (%)"]
        if isinstance(record["Confidence (%)"], (int, float))
        else None
    ))

    conn.commit()
    conn.close()


def get_history():

    conn = sqlite3.connect(DB_NAME)

    history = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY id DESC",
        conn
    )

    conn.close()

    if not history.empty:

        student_data = history["student_data"].apply(json.loads)
        student_data_df = pd.DataFrame(student_data.tolist())

        history = pd.concat([
            history.drop(columns=["student_data", "id"]),
            student_data_df
        ], axis=1)

        history = history.rename(columns={
            "date_time": "Date & Time",
            "predicted_class": "Predicted Class",
            "confidence": "Confidence (%)"
        })

    return history


def clear_history():

    conn = sqlite3.connect(DB_NAME)

    conn.execute("DELETE FROM predictions")

    conn.commit()
    conn.close()


create_database()


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
# ---------------- PLACEMENT DISTRIBUTION CHART ----------------

st.markdown("### 📊 Placement Distribution")

history = st.session_state.get("prediction_history", [])

if history:

    chart_df = pd.DataFrame(history)

    placement_counts = (
        chart_df["Predicted Class"]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    placement_counts.columns = ["Placement Status", "Students"]

    fig = px.pie(
        placement_counts,
        names="Placement Status",
        values="Students",
        title="Student Placement Distribution",
        hole=0.45
    )

    fig.update_traces(textinfo="label+percent")
    

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Make predictions to display placement charts.")

# ---------------- IQ VS CGPA PERFORMANCE ANALYSIS ----------------

st.markdown("### 📈 Student Performance Analysis")

if history:

    chart_df = pd.DataFrame(history)

    fig2 = px.scatter(
        chart_df,
        x="IQ",
        y="CGPA",
        color="Predicted Class",
        title="IQ vs CGPA Student Analysis",
        labels={
            "IQ": "Student IQ",
            "CGPA": "Student CGPA",
            "Predicted Class": "Placement Status"
        },
        hover_data=["Predicted Class"]
    )

    fig2.update_layout(
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Make predictions to display student performance analysis.")

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


# ---------------- PREDICTION HISTORY ----------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ---------------- PREDICTION ----------------

if submitted:

    input_data = pd.DataFrame(
        [input_values],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        predicted_index = list(model.classes_).index(prediction)
        confidence = probabilities[predicted_index] * 100

    # Save prediction record
    record = {
        "Date & Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        **input_values,
        "Predicted Class": str(prediction),
        "Confidence (%)": round(confidence, 2) if confidence is not None else "N/A"
    }

    st.session_state.prediction_history.append(record)

    # Display result
    st.divider()
    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Predicted Class: {prediction}")

    with col2:
        if confidence is not None:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")
            st.progress(float(confidence / 100))
        else:
            st.info("Probability is not available.")

    # Display probabilities
    if probabilities is not None:

        st.subheader("Class Probability Analysis")

        probability_df = pd.DataFrame({
            "Class": [str(c) for c in model.classes_],
            "Probability (%)": probabilities * 100
        })

        st.bar_chart(
            probability_df.set_index("Class")
        )

# ---------------- PREDICTION HISTORY TABLE ----------------

st.divider()
st.subheader("Student Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Download history
    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction History (CSV)",
        data=csv_data,
        file_name="student_placement_history.csv",
        mime="text/csv",
        use_container_width=True
    )

    if st.button("Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info("No prediction records available yet. Make your first prediction!")
    
# ---------------- FOOTER ----------------



# ---------------- FOOTER ----------------
st.divider()

st.markdown(
    "<center>Developed with Python, Scikit-learn "
    "and Streamlit | Student Placement AI Project</center>",
    unsafe_allow_html=True
)
