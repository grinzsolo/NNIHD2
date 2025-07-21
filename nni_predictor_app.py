import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, time

# โหลดโมเดลและ scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

# เชื่อม Google Sheet ผ่าน streamlit_gsheets
conn = st.connection("gsheets", type="gsheets")
existing = conn.read(worksheet="Sheet1", usecols=list(range(9)), ttl=5)

st.title("🔬 NNI Predictor with Google Sheets")

# อินพุต
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        input_date = st.date_input("Date", value=datetime.today())
        polymer_grade = st.text_input("🏷️ Polymer Grade")
    with col2:
        input_time = st.time_input("Time", value=time(hour=0,minute=0))
        user_name = st.text_input("👤 User")

    a = st.number_input("A (LC)", step=1, format="%d")
    b = st.number_input("B (MFR_S205)", step=0.1)
    c = st.number_input("C (MFR_S206)", step=0.1)
    d = st.number_input("D (MFR_S402C)", step=0.1)

    submitted = st.form_submit_button("Predict & Save")
    if submitted:
        X = np.array([[a,b,c,d]])
        pred = float(model.predict(scaler.transform(X))[0])
        st.success(f"🔮 Predicted NNI = {pred:.2f}")

        new_row = {
            "Date": input_date.strftime("%Y-%m-%d"),
            "Time": input_time.strftime("%H:%M:%S"),
            "User_Name": user_name,
            "Polymer_Grade": polymer_grade,
            "A_LC": a, "B_MFR_S205": b,
            "C_MFR_S206": c, "D_MFR_S402C": d,
            "Predicted_NNI": pred
        }

        df_new = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        conn.update(worksheet="Sheet1", data=df_new)
        st.balloons()
        st.dataframe(df_new.tail(5))
