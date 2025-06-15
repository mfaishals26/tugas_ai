# app.py

import streamlit as st
import joblib

# Load model dan vectorizer
model = joblib.load("model_random_forest.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Judul
st.title("🎯 Deteksi Emosi dalam Teks Bahasa Indonesia")
st.markdown("Masukkan kalimat untuk mengetahui emosi yang terkandung.")

# Input
teks_input = st.text_area("✏️ Tulis kalimat di sini:")

# Prediksi
if st.button("Prediksi Emosi"):
    if teks_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        teks_vec = vectorizer.transform([teks_input])
        prediksi = model.predict(teks_vec)
        st.success(f"🧠 Emosi terdeteksi: **{prediksi[0]}**")
