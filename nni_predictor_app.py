import streamlit as st
import numpy as np
import pandas as pd
import joblib
import subprocess
from datetime import datetime, time
import os
import pytz

# โหลดโมเดลและ scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
model_name = type(model).__name__

# โหลด secrets สำหรับ GitHub
gh_user = st.secrets["github"]["username"]
gh_repo = st.secrets["github"]["repo"]
gh_token = st.secrets["github"]["token"]
repo_url = f"https://{gh_token}@github.com/{gh_user}/{gh_repo}.git"

# ชื่อไฟล์ log
log_file = "prediction_log.csv"

# โหลดข้อมูลเดิม ถ้ายังไม่มีให้สร้างคอลัมน์
if os.path.exists(log_file):
    existing = pd.read_csv(log_file)
else:
    existing = pd.DataFrame(columns=[
        "Date", "Time", "User_Name", "Polymer_Grade",
        "A_LC", "B_MFR_S205", "C_MFR_S206", "D_MFR_S402C",
        "Predicted_NNI", "Log_Timestamp"
    ])

# Header UI
st.title("🔬 NNI HDPE2 Prediction 1.0")
st.markdown(f"**Model Type:** `{model_name}`")

# ----------- แบบฟอร์มอินพุต ------------
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        input_date = st.date_input("📅 Date", value=datetime.today())
        polymer_grade = st.text_input("🏷️ Polymer Grade", placeholder="เช่น HD7000F")
    with col2:
        input_time = st.time_input("⏰ Time", value=time(hour=0, minute=0))
        user_name = st.text_input("👤 User", placeholder="เช่น Parom W.")

    # ค่าอินพุตตัวเลข
    a = st.number_input("🧪 A (LC)", step=1, format="%d", min_value=1)
    b = st.number_input("🧪 B (MFR_S205)", step=0.1, format="%.2f", min_value=0.01)
    c = st.number_input("🧪 C (MFR_S206)", step=0.1, format="%.2f", min_value=0.01)
    d = st.number_input("🧪 D (MFR_S402C)", step=0.1, format="%.2f", min_value=0.01)

    submitted = st.form_submit_button("✅ Predict & Save")

    if submitted:
        # ตรวจสอบความครบถ้วนของข้อมูล
        missing_fields = []
        if polymer_grade.strip() == "":
            missing_fields.append("Polymer Grade")
        if user_name.strip() == "":
            missing_fields.append("User")

        if missing_fields:
            st.warning("⚠️ กรุณากรอกข้อมูลต่อไปนี้ให้ครบ: " + ", ".join(missing_fields))
        else:
            # พยากรณ์
            X = np.array([[a, b, c, d]])
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled)[0])

            st.success(f"🔮 Predicted NNI = `{pred:.2f}`")

            # เวลาไทย (UTC+7)
            thai_time = datetime.now(pytz.timezone("Asia/Bangkok"))
            log_ts = thai_time.strftime("%Y-%m-%d %H:%M:%S")

            # สร้างแถวใหม่
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
                "Log_Timestamp": log_ts
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

            # แสดงผลตารางล่าสุด
            st.markdown("### 🧾 Recent Logs")
            st.dataframe(updated_df.tail(5))
