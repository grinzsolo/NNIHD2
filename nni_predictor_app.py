import streamlit as st
import numpy as np
import pandas as pd
import joblib
import subprocess
from datetime import datetime, time
import os
import pytz  # ✅ เพิ่ม pytz สำหรับ timezone

# โหลดโมเดลและ scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
model_name = type(model).__name__

# โหลด secrets
gh_user = st.secrets["github"]["username"]
gh_repo = st.secrets["github"]["repo"]
gh_token = st.secrets["github"]["token"]
repo_url = f"https://{gh_token}@github.com/{gh_user}/{gh_repo}.git"

# ชื่อไฟล์ CSV log
log_file = "prediction_log.csv"

# โหลดข้อมูลเดิม
if os.path.exists(log_file):
    existing = pd.read_csv(log_file)
else:
    existing = pd.DataFrame(columns=[
        "Date", "Time", "User_Name", "Polymer_Grade",
        "A_LC", "B_MFR_S205", "C_MFR_S206", "D_MFR_S402C",
        "Predicted_NNI", "Log_Timestamp"  # ✅ เพิ่ม column
    ])

st.title("🔬 NNI HDPE2 Prediction 1.0")
st.markdown(f"**Model Type:** `{model_name}`")

# -------- ฟอร์มอินพุต --------
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        input_date = st.date_input("📅 Sample Date", value=datetime.today())
        polymer_grade = st.text_input("🏷️ Polymer Grade", placeholder="เช่น HD7000F")
    with col2:
        input_time = st.time_input("⏰ Sample Time", value=time(hour=0, minute=0))
        user_name = st.text_input("👤 User", placeholder="เช่น Parom W.")

    a = st.number_input("🧪 A (LC)", step=1, format="%d")
    b = st.number_input("🧪 B (MFR_S205)", step=0.001, format="%.3f")
    c = st.number_input("🧪 C (MFR_S206)", step=0.001, format="%.3f")
    d = st.number_input("🧪 D (MFR_S402C)", step=0.001, format="%.3f")

    submitted = st.form_submit_button("✅ Predict & Save")

    if submitted:
        if polymer_grade.strip() == "" or user_name.strip() == "":
            st.warning("กรุณากรอก Polymer Grade และ User ให้ครบ")
        else:
            X = np.array([[a, b, c, d]])
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled)[0])

            st.success(f"🔮 Predicted NNI = `{pred:.2f}`")

            # ✅ สร้าง timestamp เวลาไทย
            thai_time = datetime.now(pytz.timezone("Asia/Bangkok"))
            log_ts = thai_time.strftime("%Y-%m-%d %H:%M:%S")

            new_row = {
                "Date": input_date.strftime("%Y-%m-%d"),
                "Time": input_time.strftime("%H:%M:%S"),
                "User_Name": user_name,
                "Polymer_Grade": polymer_grade,
                "A_LC": a,
                "B_MFR_S205": b,
                "C_MFR_S206": c,
                "D_MFR_S402C": d,
                "Predicted_NNI": pred,
                "Log_Timestamp": log_ts  # ✅ เพิ่ม timestamp column
            }

            updated_df = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
            updated_df.to_csv(log_file, index=False)

            # Git Commit & Push
            try:
                subprocess.run(["git", "config", "--global", "user.email", f"{gh_user}@users.noreply.github.com"], check=True)
                subprocess.run(["git", "config", "--global", "user.name", gh_user], check=True)
                subprocess.run(["git", "add", log_file], check=True)
                subprocess.run(["git", "commit", "-m", "📈 New prediction entry added"], check=True)
                subprocess.run(["git", "push", repo_url], check=True)

                st.success("📤 Log uploaded to GitHub!")
            except subprocess.CalledProcessError as e:
                st.error("❌ Git error: " + str(e))

            st.dataframe(updated_df.tail(5))
