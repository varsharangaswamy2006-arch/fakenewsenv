import streamlit as st
import requests
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Fake News Detector", layout="wide")

# 👉 Change this if deployed
API_URL = "http://localhost:7860"

st.title("🧠 Fake News Detector (OpenEnv Ready)")

# =========================
# INPUT
# =========================
text = st.text_area("Enter News Text")

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": text}
            )

            if response.status_code == 200:
                data = response.json()

                label = data["label"]
                confidence = data["confidence"]

                # =========================
                # OUTPUT
                # =========================
                if label == 1:
                    st.success("🟢 REAL NEWS")
                else:
                    st.error("🔴 FAKE NEWS")

                st.write(f"Confidence: {round(confidence, 3)}")

                # =========================
                # VISUALIZATION
                # =========================
                fake_prob = 1 - confidence
                real_prob = confidence

                fig, ax = plt.subplots()
                ax.bar(["FAKE", "REAL"], [fake_prob, real_prob])
                ax.set_ylabel("Probability")

                st.pyplot(fig)

            else:
                st.error("API Error")

        except Exception as e:
            st.error("⚠️ Backend not running. Make sure inference.py is running!")
