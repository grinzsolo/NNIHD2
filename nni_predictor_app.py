import streamlit as st
import numpy as np
import pandas as pd
import joblib
import subprocess
from datetime import datetime, time
import os
import pytz
import requests
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, FollowEvent, TextSendMessage
import json

# -------- LINE Bot setup --------
LINE_CHANNEL_ACCESS_TOKEN = st.secrets["line_messaging"]["access_token"]
LINE_CHANNEL_SECRET = st.secrets["line_messaging"]["channel_secret"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# -------- Flask app for webhook --------
app = Flask(__name__)

USER_FILE = "line_user_ids.json"

def load_user_ids():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return []

def save_user_id(user_id):
    user_ids = load_user_ids()
    if user_id not in user_ids:
        user_ids.append(user_id)
        with open(USER_FILE, "w") as f:
            json.dump(user_ids, f)

def send_line_message(user_id: str, message: str):
    access_token = LINE_CHANNEL_ACCESS_TOKEN
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    body = {
        "to": user_id,
        "messages": [{
            "type": "text",
            "text": message
        }]
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code != 200:
        st.warning(f"⚠️ LINE Messaging API error: {response.text}")

def broadcast_message(text: str):
    user_ids = load_user_ids()
    for uid in user_ids:
        send_line_message(uid, text)

# -------- Flask webhook routes --------
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(FollowEvent)
def handle_follow(event):
    user_id = event.source.user_id
    save_user_id(user_id)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="ขอบคุณที่เพิ่มบอทครับ! คุณจะได้รับแจ้งเตือนจากระบบ")
    )

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_id = event.source.user_id
    save_user_id(user_id)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="ขอบคุณสำหรับข้อความครับ!")
    )

# -------- Load model and scaler --------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
model_name = type(model).__name__

gh_user = st.secrets["github"]["username"]
gh_repo = st.secrets["github"]["repo"]
gh_token = st.secrets["github"]["token"]
repo_url = f"https://{gh_token}@github.com/{gh_user}/{gh_repo}.git"

log_file = "prediction_log.csv"
if os.path.exists(log_file):
    existing = pd.read_csv(log_file)
else:
    existing = pd.DataFrame(columns=[
        "Date", "Time", "User_Name", "Polymer_Grade",
        "A_LC", "B_MFR_S205", "C_MFR_S206", "D_MFR_S402C",
        "Predicted_NNI", "Log_Timestamp"
    ])

# -------- Streamlit UI --------
st.title("🔬 NNI HDPE2 Prediction 1.0")
st.markdown(f"**Model Type:** `{model_name}`")

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

                # Broadcast แจ้งเตือนทุก user ใน list
                line_msg = (
                    f"🔔 New NNI Prediction\n"
                    f"👤 User: {user_name}\n"
                    f"📅 Date: {input_date.strftime('%Y-%m-%d')} {input_time.strftime('%H:%M')}\n"
                    f"🏷️ Grade: {polymer_grade}\n"
                    f"🧪 Inputs: LC={a}, S205={b}, S206={c}, S402C={d}\n"
                    f"🔮 Predicted NNI: {pred:.2f}"
                )
                broadcast_message(line_msg)

            except subprocess.CalledProcessError as e:
                st.error("❌ Git error: " + str(e))

            st.dataframe(updated_df.tail(5))

# -------- Run Flask webhook server (optional) --------
if __name__ == "__main__":
    import threading

    def run_flask():
        app.run(port=8000)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
