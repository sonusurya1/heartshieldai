import streamlit as st
import numpy as np
import pickle
from fpdf import FPDF

# ---------------- LOAD MODEL & SCALER ----------------
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- SAFE TEXT FUNCTION (UNICODE FIX) ----------------
def clean_text(text):
    return text.encode("latin-1", "ignore").decode("latin-1")

# ---------------- PDF GENERATOR ----------------
def generate_pdf(age, sex, BP, cholesterol, heart_rate, smoking, risk, prediction, medicines):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)

    pdf.cell(0, 10, clean_text("HeartShield AI - Medical Report"), ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, clean_text(f"Age: {age}"), ln=True)
    pdf.cell(0, 10, clean_text(f"Sex: {'Male' if sex==1 else 'Female'}"), ln=True)
    pdf.cell(0, 10, clean_text(f"Blood Pressure: {'High' if BP==1 else 'Normal'}"), ln=True)
    pdf.cell(0, 10, clean_text(f"Cholesterol: {'High' if cholesterol==1 else 'Normal'}"), ln=True)
    pdf.cell(0, 10, clean_text(f"Resting Heart Rate: {heart_rate}"), ln=True)
    pdf.cell(0, 10, clean_text(f"Smoking: {'Yes' if smoking==1 else 'No'}"), ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, clean_text(f"Heart Disease Risk Probability: {risk:.2f} percent"), ln=True)

    result = "High Risk" if prediction == 1 else "Low Risk"
    pdf.cell(0, 10, clean_text(f"Prediction Result: {result}"), ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, clean_text("Recommended Medicines:"), ln=True)

    pdf.set_font("Arial", size=11)
    for med in medicines:
        pdf.cell(0, 8, clean_text(f"- {med}"), ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 8, clean_text("Disclaimer: This is an AI-based suggestion. Consult a doctor."), ln=True)

    pdf.output("HeartShield_Report.pdf")

# ---------------- STREAMLIT UI ----------------
st.title("HeartShield AI - Heart Disease Prediction")
st.write("Enter patient details to predict heart disease risk")

age = st.number_input("Enter Age", min_value=1, max_value=120, value=26)
sex_input = st.selectbox("Enter Sex", ["Male", "Female"])
BP_input = st.selectbox("Blood Pressure", ["Normal", "High"])
chol_input = st.selectbox("Cholesterol Level", ["Normal", "High"])
heart_rate = st.number_input("Resting Heart Rate", min_value=40, max_value=200, value=90)
smoking_input = st.selectbox("Do you Smoke?", ["No", "Yes"])

# ---------------- ENCODING ----------------
sex = 1 if sex_input == "Male" else 0
BP = 1 if BP_input == "High" else 0
cholesterol = 1 if chol_input == "High" else 0
smoking = 1 if smoking_input == "Yes" else 0

# ---------------- PREDICTION ----------------
if st.button("Predict Heart Disease"):
    input_data = np.array([[age, sex, BP, cholesterol, heart_rate, smoking]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

    st.info(f"Risk Probability: {probability:.2f} percent")

    # ---------------- MEDICINE RECOMMENDATION ----------------
    st.subheader("Medicine Recommendation")
    medicines = []

    if BP == 1:
        medicines.append("Amlodipine (for High Blood Pressure)")
    if cholesterol == 1:
        medicines.append("Atorvastatin (for High Cholesterol)")
    if smoking == 1:
        medicines.append("Nicotine Replacement Therapy")
    if prediction == 1:
        medicines.append("Low-dose Aspirin (Doctor Consultation Required)")
    else:
        medicines.append("No medicine required. Maintain healthy lifestyle.")

    for med in medicines:
        st.write("- ", med)

    # ---------------- PDF DOWNLOAD ----------------
    generate_pdf(age, sex, BP, cholesterol, heart_rate, smoking, probability, prediction, medicines)

    with open("HeartShield_Report.pdf", "rb") as file:
        st.download_button(
            label="Download Medical PDF Report",
            data=file,
            file_name="HeartShield_Report.pdf",
            mime="application/pdf"
        )
