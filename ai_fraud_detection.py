import streamlit as st
from email import policy
from email.parser import BytesParser
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import sqlite3
import os
import csv
import json

import fraud_core as fc

# ---------- OCR PATH FIX (WINDOWS ONLY) ----------
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# ---------- DEMO LOGIN (override with env vars / Streamlit secrets) ----------
VALID_USERNAME = os.getenv("APP_USERNAME", "admin")
VALID_PASSWORD = os.getenv("APP_PASSWORD", "1234")

# ---------- DATABASE SETUP ----------
DB_FILE = "fraud_detection.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_type TEXT,
            content TEXT,
            classification TEXT,
            confidence INTEGER,
            reason TEXT,
            recommendation TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_analysis(input_type, content, classification, confidence, reason, recommendation):
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''
        INSERT INTO analysis (input_type, content, classification, confidence, reason, recommendation)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (input_type, content, classification, confidence, reason, recommendation))
    conn.commit()
    conn.close()

CSV_FILE = "fraud_detection_data.csv"

def save_to_csv(input_type, content, classification, confidence, reason, recommendation):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Input Type", "Content", "Classification", "Confidence", "Reason", "Recommendation"])
        writer.writerow([input_type, content, classification, confidence, reason, recommendation])

# ---------- SESSION STATE (login is per-browser-session now) ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "history" not in st.session_state:
    st.session_state.history = []
if "input_usage" not in st.session_state:
    st.session_state.input_usage = {"Text": 0, "Email (.eml)": 0, "Image": 0}

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("🔐 Login")
    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    else:
        st.success(f"👋 Logged in as {VALID_USERNAME}")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    if st.session_state.logged_in:
        st.divider()
        st.header("⚙️ Detection Mode")
        mode = st.radio(
            "Detector",
            ["Hybrid (LLM + rules)", "Rules only (free)", "LLM only"],
            help="Rules only needs no API key and never hits quota limits."
        )

# ---------- MAIN APP ----------
if st.session_state.logged_in:
    st.title("🚨 AI Fraud Detection Dashboard 🚨")
    st.write("Detects fraud in **Text, Email, and Images**")
    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("📥 Input Section")
        input_type = st.selectbox("Select Input Type", ["Text", "Email (.eml)", "Image"])
        extracted_text = ""

        if input_type == "Text":
            extracted_text = st.text_area("Enter message text")

        elif input_type == "Email (.eml)":
            uploaded_email = st.file_uploader("Upload .eml file", type=["eml"])
            if uploaded_email:
                msg = BytesParser(policy=policy.default).parse(uploaded_email)
                body = msg.get_body(preferencelist=("plain", "html"))
                if body:
                    extracted_text = body.get_content()

        elif input_type == "Image":
            uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                try:
                    extracted_text = pytesseract.image_to_string(image)
                except Exception as e:
                    st.error(f"OCR failed (is Tesseract installed?): {e}")

        if extracted_text:
            st.subheader("📤 Extracted Text")
            st.write(extracted_text)

        detect = st.button("🚀 Detect Fraud")

    with right:
        st.subheader("📊 Live Analytics")
        if st.session_state.history:
            counts = {k: st.session_state.history.count(k) for k in fc.LABELS}
            st.bar_chart(counts)

        if sum(st.session_state.input_usage.values()) > 0:
            fig, ax = plt.subplots()
            ax.pie(
                st.session_state.input_usage.values(),
                labels=st.session_state.input_usage.keys(),
                autopct="%1.1f%%",
            )
            ax.set_title("Input Type Usage")
            st.pyplot(fig)

    # ================= DETECTION =================
    if detect and extracted_text.strip():
        st.session_state.input_usage[input_type] += 1

        if mode.startswith("Rules"):
            classification, confidence, reason = fc.rules_classify(extracted_text)
            source = "rules"
        elif mode.startswith("LLM"):
            try:
                classification, confidence, reason = fc.llm_classify(extracted_text)
                source = "llm"
            except fc.LLMUnavailable as e:
                st.error(f"LLM unavailable: {e}  → switch to 'Rules only' or 'Hybrid' in the sidebar.")
                st.stop()
        else:
            classification, confidence, reason, source = fc.hybrid_classify(extracted_text)

        if source.startswith("rules (LLM"):
            st.warning(f"⚠️ {source.replace('rules', 'Used rules only', 1)}")

        st.session_state.history.append(classification)

        if classification == "Fraud":
            recommendation = "⚠️ Do not click suspicious links or share personal info. Report this to authorities."
        elif classification == "Spam":
            recommendation = "🚫 Avoid replying or engaging with spam content. Block the sender if possible."
        elif classification == "Legit":
            recommendation = "✅ Message seems safe. Always double-check unknown sources."
        else:
            recommendation = "❓ Could not classify this message. Treat it with caution."

        save_analysis(input_type, extracted_text, classification, confidence, reason, recommendation)
        save_to_csv(input_type, extracted_text, classification, confidence, reason, recommendation)

        st.divider()
        st.subheader("🤖 AI Result")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Classification:** {classification}")
            st.write(f"**Reason:** {reason}")
            st.caption(f"Detector used: {source}")
            st.progress(confidence / 100)
        with col2:
            st.metric("Confidence Score", f"{confidence}%")
            if classification == "Fraud":
                st.error("🔴 Fraud detected" if confidence >= 80 else "🟡 Likely fraud")
            elif classification == "Spam":
                st.warning("🟡 Spam")
            elif classification == "Legit":
                st.success("🟢 Looks legitimate")
            else:
                st.info("❓ Unknown")
        st.info(f"💡 **Recommendation:** {recommendation}")

    # ---------- MODEL EVALUATION (from evaluate.py) ----------
    st.divider()
    st.subheader("🧪 Model Evaluation")
    if os.path.exists("evaluation_summary.json"):
        with open("evaluation_summary.json") as f:
            ev = json.load(f)
        cols = st.columns(len(ev["modes"]))
        for c, (name, m) in zip(cols, ev["modes"].items()):
            c.metric(f"{name.capitalize()} accuracy", f"{m['accuracy']*100:.1f}%",
                     help=f"{m['samples']} labeled test messages, macro-F1 {m['macro_f1']}")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", caption=f"Confusion matrix ({ev['best_mode']})", width=450)
    else:
        st.caption("Run `python evaluate.py --mode all` to see accuracy metrics here.")

    # ---------- HISTORY VIEW ----------
    st.divider()
    st.subheader("📚 Analysis History (from Database)")
    conn = sqlite3.connect(DB_FILE)
    try:
        data = conn.execute(
            "SELECT id, input_type, classification, confidence, recommendation, timestamp "
            "FROM analysis ORDER BY id DESC"
        ).fetchall()
        st.dataframe(data, use_container_width=True)
    except Exception as e:
        st.error(f"Database error: {e}")
    conn.close()

    st.subheader("📦 Export Data")
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.download_button("⬇️ Download CSV Data", f, file_name="fraud_detection_data.csv", mime="text/csv")

else:
    st.warning("🔒 Please log in from the sidebar to access the Fraud Detection Dashboard.")
