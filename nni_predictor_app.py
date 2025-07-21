import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time

# โหลด model และ scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# อ่าน/เขียน Google Sheet ผ่าน streamlit_gsheets
conn = st.connection("gsheets", type="gsheets")
existing_data = conn.read(worksheet="Sheet1", usecols=list(range(9)), ttl=5)

st.title("🔬 NNI Predictor with Google Sheets")

# รับ input
col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Date", value=datetime.today())
    polymer_grade = st.text_input("🏷️ Polymer Grade")
with col2:
    input_time = st.time_input("Time", value=time(hour=0, minute=0))
    user_name = st.text_input("👤 User")

a = st.number_input("Input A (LC)", step=1, format="%d")
b = st.number_input("Input B (MFR_S205)", step=0.1)
c = st.number_input("Input C (MFR_S206)", step=0.1)
d = st.number_input("Input D (MFR_S402C)", step=0.1)

# ทำนายและบันทึก
if st.button("Predict and Save"):
    if polymer_grade.strip() == "" or user_name.strip() == "":
        st.warning("⚠️ กรุณากรอก Polymer Grade และ User Name")
    elif any(v == 0 for v in [a, b, c, d]):
        st.warning("⚠️ กรุณาใส่ค่าทุก parameter")
    else:
        X = np.array([[a, b, c, d]])
        X_scaled = scaler.transform(X)
        prediction = float(model.predict(X_scaled)[0])
        st.success(f"Predicted NNI = {prediction:.2f}")

        new_row = {
            "Date": input_date.strftime("%Y-%m-%d"),
            "Time": input_time.strftime("%H:%M:%S"),
            "User_Name": user_name,
            "Polymer_Grade": polymer_grade,
            "A_LC": a,
            "B_MFR_S205": b,
            "C_MFR_S206": c,
            "D_MFR_S402C": d,
            "Predicted_NNI": prediction
        }

        df_new = pd.DataFrame([new_row])
        updated_data = pd.concat([existing_data, df_new], ignore_index=True)

        conn.update(worksheet="Sheet1", data=updated_data)
        st.info("✅ บันทึกลง Google Sheet สำเร็จ")
        st.dataframe(updated_data.tail(5))
