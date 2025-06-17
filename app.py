import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import os
import gdown
import zipfile

# ========== CONFIG ========== #
st.set_page_config(page_title="Deteksi Emosi AI", page_icon="💬", layout="wide")

# ========== CUSTOM CSS & EMOJI BACKGROUND ANIMATION ========== #
st.markdown("""
    <style>
    body { background-color: #121212; color: #f0f0f0; }
    .stApp { background-color: #1e1e1e; }
    .stButton>button {
        background-color: #00c896; color: white; font-weight: bold;
        border-radius: 8px; padding: 10px 20px; border: none;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #2c2c2c; color: white; border-radius: 8px;
    }
    .stSelectbox>div>div {
        background-color: #2c2c2c !important; color: white !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #00ffc3; }
    .emojiRain {
        font-size: 2rem;
        animation: fall 5s infinite;
        position: fixed;
        z-index: 0;
    }
    @keyframes fall {
        0% { top: -10%; opacity: 0; }
        50% { opacity: 1; }
        100% { top: 110%; opacity: 0; }
    }
    </style>
    <div class="emojiRain" style="left:10%;">😊</div>
    <div class="emojiRain" style="left:25%;">😢</div>
    <div class="emojiRain" style="left:40%;">😠</div>
    <div class="emojiRain" style="left:55%;">😌</div>
    <div class="emojiRain" style="left:70%;">😣</div>
""", unsafe_allow_html=True)

# ========== LABEL EMOSI ========== #
labels = {
    0: "Bersyukur",
    1: "Marah",
    2: "Sedih",
    3: "Senang",
    4: "Stress",
}

# ========== LOAD MODEL DARI GOOGLE DRIVE (ZIP) ========== #
@st.cache_resource
def load_model():
    model_dir = "final_model"
    zip_file = "final_model.zip"
    file_id = "1htLjHXU3kQchKbvqMumQ9iW95EjDY2M0"  # GANTI dengan ID ZIP kamu
    url = f"https://drive.google.com/file/d/1htLjHXU3kQchKbvqMumQ9iW95EjDY2M0/view?usp=sharing{file_id}"

    if not os.path.exists(model_dir):
        with st.spinner("📦 Mengunduh dan mengekstrak model..."):
            gdown.download(url, zip_file, quiet=False)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall()
            os.remove(zip_file)

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()
model.eval()

# ========== FUNGSI PREDIKSI ========== #
def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
    predicted_label = labels[np.argmax(probs)]
    return predicted_label, probs

# ========== SIDEBAR ========== #
st.sidebar.title("🧭 Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Beranda", "🧠 Deteksi Emosi", "📑 Deteksi Massal", "ℹ️ Tentang"])
st.sidebar.markdown("---")
st.sidebar.info("✨ Powered by IndoBERT\n👨‍💻 Kelompok 1")

# ========== BERANDA ========== #
if menu == "🏠 Beranda":
    st.title("💬 Deteksi Emosi Teks Bahasa Indonesia")
    st.markdown("""
        Aplikasi ini menggunakan **AI IndoBERT** untuk mendeteksi emosi dari kalimat berbahasa Indonesia.

        ### 🎯 Fitur:
        - Deteksi emosi pada satu atau banyak kalimat
        - Visualisasi dengan pie chart
        - Ekspor hasil ke CSV
    """)

# ========== DETEKSI EMOSI ========== #
elif menu == "🧠 Deteksi Emosi":
    st.title("🔍 Deteksi Emosi dari Kalimat")

    contoh_kalimat = st.selectbox("📋 Pilih Contoh Kalimat", (
        "",  # Default kosong
        "Aku senang banget hari ini dapet kabar baik!",
        "Gue marah banget ama dia!",
        "Sedih banget rasanya ditinggal pas sayang-sayangnya.",
        "Stress banget mau UTS tapi belum belajar.",
        "Biasa aja sih, ngga terlalu penting."
    ))

    if contoh_kalimat:
        st.session_state["isi_otomatis"] = contoh_kalimat

    user_input = st.text_area("✍️ Masukkan Teks Kamu", value=st.session_state.get("isi_otomatis", ""), placeholder="Contoh:\nAku senang dapet nilai bagus!")

    if st.button("🚀 Deteksi Sekarang"):
        if user_input.strip():
            with st.spinner("🔍 Mendeteksi emosi... ✨😊😢😠"):
                label, probas = predict_emotion(user_input)
            prob_dict = {labels[i]: float(probas[i]) for i in range(len(labels))}
            st.success(f"💡 Emosi Terdeteksi: **{label}**")

            fig = px.pie(
                names=list(prob_dict.keys()),
                values=list(prob_dict.values()),
                title="Distribusi Probabilitas Emosi",
                color_discrete_sequence=px.colors.sequential.Agsunset
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📥 Simpan Hasil"):
                hasil_df = pd.DataFrame({
                    "Teks": [user_input],
                    "Emosi": [label],
                    "Probabilitas": [str(prob_dict)]
                })
                st.download_button("📥 Unduh CSV", data=hasil_df.to_csv(index=False), file_name="hasil_klasifikasi.csv", mime="text/csv")
        else:
            st.warning("Teks tidak boleh kosong!")

# ========== DETEKSI MASSAL ========== #
elif menu == "📑 Deteksi Massal":
    st.title("📋 Deteksi Emosi Massal")
    teks_massal = st.text_area("📝 Masukkan beberapa kalimat (1 baris 1 kalimat):", height=200, placeholder="Contoh:\nAku senang dapet nilai bagus!\nAku kesel banget ama dia!")

    if st.button("🚀 Jalankan Deteksi Massal"):
        if teks_massal.strip():
            kalimat_list = teks_massal.strip().splitlines()
            hasil_massal = []
            for kalimat in kalimat_list:
                emosi, _ = predict_emotion(kalimat)
                hasil_massal.append({"Teks": kalimat, "Emosi": emosi})
            df_massal = pd.DataFrame(hasil_massal)
            st.dataframe(df_massal)
            st.download_button("💾 Unduh Hasil", data=df_massal.to_csv(index=False), file_name="hasil_massal.csv", mime="text/csv")
        else:
            st.warning("Masukkan setidaknya satu kalimat.")

# ========== TENTANG ========== #
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi Ini")
    st.markdown("""
    Aplikasi deteksi emosi ini dibuat dengan:

    - 🤗 Transformers (IndoBERT)
    - 🔥 PyTorch
    - 🖥️ Streamlit modern style

    Dibuat untuk tugas akhir kuliah **Kecerdasan Buatan**.

    👨‍💻 Developer: **Kelompok 1**
    """)
    st.caption("© 2025 | Sistem Deteksi Emosi Bahasa Indonesia")
