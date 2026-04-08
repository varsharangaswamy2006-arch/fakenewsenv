import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from inference import predict  # reuse model

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("🧠 Fake News Detector")

text = st.text_area("Enter News")

if st.button("Predict"):
    label, confidence, prob = predict(text)

    st.success("REAL" if label else "FAKE")
    st.write("Confidence:", round(confidence, 3))

    fig, ax = plt.subplots()
    ax.bar(["FAKE", "REAL"], prob)
    st.pyplot(fig)
