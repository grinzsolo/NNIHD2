import streamlit as st
import numpy as np
import joblib
from datetime import datetime, time
from gspread_pandas import Spread, Client
import pandas as pd

# ---------------- CONFIG ----------------
SPREADSHEET_URL = st.secrets["SPREADSHEET_URL"]  # ควรเก็บใน secrets.toml
SHEET_NAME = "Sheet1"  # หรือชื่อที่คุณตั้งไว้ใน Google Sheet

# เชื่อมต่อแบบ anonymous (public sheet เท่านั้น)
spread = Spread(spread=SPREADSHEET_URL)

# โหลดโมเดล
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔬 NNI Predictor with Public Google Sheet")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Date", value=datetime.today())
    polymer_grade = st.text_input("🏷️ Polymer Grade", placeholder="เช่น HD7000F")
with col2:
    input_time = st.time_input("Time", value=time(hour=0, minute=0))
    user_name = st.text_input("👤 User", placeholder="เช่น Parom W.")

a = st.number_input("Input A (LC)", step=1, format="%d")
b = st.number_input("Input B (MFR_S205)", step=0.1, format="%.2f")
c = st.number_input("Input C (MFR_S206)", step=0.1, format="%.2f")
d = st.number_input("Input D (MFR_S402C)", step=0.1, format="%.2f")

# ---------------- PREDICT ----------------
if st.button("Predict and Save to Google Sheet"):
    if polymer_grade.strip() == "" or user_name.strip() == "":
        st.warning("⚠️ กรุณากรอก Polymer Grade และ User Name ให้ครบ")
    elif any(v is None or v == 0 for v in [a, b, c, d]):
        st.warning("⚠️ กรุณากรอกค่าทุกพารามิเตอร์ A, B, C, D")
    else:
        X = np.array([[a, b, c, d]])
        X_scaled = scaler.transform(X)
        prediction = float(model.predict(X_scaled)[0])
        st.success(f"Predicted NNI = {prediction:.2f}")

        # เตรียมข้อมูลสำหรับ Google Sheet
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

        try:
            # อ่านข้อมูลเดิม
            df_existing = spread.sheet_to_df(sheet=SHEET_NAME, index=None)
        except Exception:
            # ถ้า Google Sheet ว่าง
            df_existing = pd.DataFrame()

        df_new = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)

        try:
            spread.df_to_sheet(df_new, sheet=SHEET_NAME, index=False)
            st.info("✅ บันทึกข้อมูลลง Google Sheet แล้ว")
        except Exception as e:
            st.error(f"❌ ไม่สามารถเขียนข้อมูลลง Google Sheet: {e}")
