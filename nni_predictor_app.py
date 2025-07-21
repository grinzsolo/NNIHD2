import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, time

# โหลด model และ scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
model_name = type(model).__name__

st.set_page_config(page_title="🧪 NNI Predictor App", layout="centered")
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; font-family: 'Segoe UI', sans-serif; }
        .stTextInput>div>div>input { background-color: #fffbe6; }
        .stNumberInput>div>div>input { background-color: #fffbe6; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 NNI Predictor Web App")
st.markdown(f"**Model Type:** `{model_name}`")
st.markdown("### ใส่ข้อมูลเพื่อทำนายค่า Output พร้อมบันทึก")
st.markdown("---")

# --------- Section 1: General Info Inputs ---------
col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Date", value=datetime.today())
    polymer_grade = st.text_input("🏷️ Polymer Grade", placeholder="เช่น HD7000F")
with col2:
    input_time = st.time_input("Time", value=time(hour=0, minute=0))
    user_name = st.text_input("👤 User", placeholder="เช่น Parom W.")

st.markdown("---")

# --------- Section 2: Model Feature Inputs ---------
a = st.number_input("Input A (LC)", step=1, format="%d")
b = st.number_input("Input B (MFR_S205)", step=0.1, format="%.2f")
c = st.number_input("Input C (MFR_S206)", step=0.1, format="%.2f")
d = st.number_input("Input D (MFR_S402C)", step=0.1, format="%.2f")

# --------- Section 3: Predict and Save ---------
if st.button("Predict and Save"):
    if polymer_grade.strip() == "" or user_name.strip() == "":
        st.warning("⚠️ กรุณากรอก Polymer Grade และ User Name ให้ครบถ้วนก่อนทำนาย")
    elif any(v is None or v == 0 for v in [a, b, c, d]):
        st.warning("⚠️ กรุณากรอกค่าทุกพารามิเตอร์ A, B, C, D ก่อนทำนาย")
    else:
        # 1. Predict
        X = np.array([[a, b, c, d]])
        X_scaled = scaler.transform(X)
        prediction = float(model.predict(X_scaled)[0])

        # 2. Show Result
        st.success(f"Predicted NNI = {prediction:.2f}")

        # 3. Prepare Record
        record = {
            "Date":           input_date.strftime("%Y-%m-%d"),
            "Time":           input_time.strftime("%H:%M:%S"),
            "User_Name":      user_name,
            "Polymer_Grade":  polymer_grade,
            "A_LC":           a,
            "B_MFR_S205":     b,
            "C_MFR_S206":     c,
            "D_MFR_S402C":    d,
            "Predicted_NNI":  prediction
        }

        # 4. Save to Google Sheets via st.connection
        conn = st.connection("gsheets", type="gspread")
        existing_df = conn.read(worksheet="Sheet1", usecols=list(record.keys()), ttl=0)
        updated_df = pd.concat([existing_df, pd.DataFrame([record])], ignore_index=True)

        # เขียนกลับไปยัง Sheet
        conn.update(worksheet="Sheet1", data=updated_df)

        # 5. Notify user
        st.info("✅ ข้อมูลถูกบันทึกลง Google Sheet แล้ว")

        # 6. Show last 5 records
        st.dataframe(updated_df.tail(5))
