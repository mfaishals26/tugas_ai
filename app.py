import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import random

# ========== CONFIG ========== #
st.set_page_config(page_title="Deteksi Emosi AI", page_icon="💬", layout="wide")

# ========== CUSTOM CSS ========== #
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
    </style>
""", unsafe_allow_html=True)

# ========== LABEL EMOSI ========== #
labels = {
    0: "Bersyukur",
    1: "Marah",
    2: "Sedih",
    3: "Senang",
    4: "Stress",
}

# ========== KUTIPAN BERDASARKAN EMOSI ========== #
quotes = {
    "Bersyukur": [
        "Rasa syukur mengubah apa yang kita miliki menjadi cukup. 🌼",
        "Bahagia itu sederhana, yaitu bersyukur. 🤲"
    ],
    "Marah": [
        "Marah hanya akan membakar hatimu sendiri. Tenangkan pikiranmu. 🔥🧊",
        "Tahan amarah, karena kamu lebih kuat dari emosimu. 💪"
    ],
    "Sedih": [
        "Kesedihan adalah bagian dari proses menjadi kuat. 💧",
        "Tidak apa-apa merasa sedih, itu tanda kamu manusia. 🤍"
    ],
    "Senang": [
        "Nikmati setiap momen bahagia. Kamu pantas mendapatkannya! 😄",
        "Kebahagiaan itu menular, bagikanlah! ✨"
    ],
    "Stress": [
        "Tarik napas, kamu sudah melakukan yang terbaik. 🌿",
        "Luangkan waktu untuk dirimu sendiri. 💆‍♂️"
    ]
}

def get_emotion_message(label):
    return random.choice(quotes.get(label, ["Tetap semangat!"]))

# ========== LOAD MODEL ========== #
@st.cache_resource
def load_model():
    repo = "faishal26/final_model"
    model = BertForSequenceClassification.from_pretrained(repo)
    tokenizer = BertTokenizer.from_pretrained(repo)
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
menu = st.sidebar.radio("Pilih Halaman", ["🏠 Beranda", "🧠 Deteksi Emosi", "📑 Deteksi Massal", "💬 Konsultasi Live", "ℹ️ Tentang"])
st.sidebar.markdown("---")
st.sidebar.info("✨ Powered by IndoBERT\n👨‍💻 Kelompok 1")

# ========== ANIMASI EMOJI ========== #
def emoji_rain_animation():
    st.markdown("""
    <div class="emoji-rain">
        <span>😊</span><span>😢</span><span>😠</span><span>🥰</span><span>😎</span><span>🤯</span><span>😭</span>
    </div>
    <style>
    .emoji-rain {
        position: relative;
        height: 50px;
        overflow: visible;
    }
    .emoji-rain span {
        position: absolute;
        animation: fall 5s linear infinite;
        font-size: 2rem;
        opacity: 0.8;
    }
    .emoji-rain span:nth-child(1) { left: 5%; animation-delay: 0s; }
    .emoji-rain span:nth-child(2) { left: 20%; animation-delay: 0.5s; }
    .emoji-rain span:nth-child(3) { left: 35%; animation-delay: 1s; }
    .emoji-rain span:nth-child(4) { left: 50%; animation-delay: 1.5s; }
    .emoji-rain span:nth-child(5) { left: 65%; animation-delay: 2s; }
    .emoji-rain span:nth-child(6) { left: 80%; animation-delay: 2.5s; }
    .emoji-rain span:nth-child(7) { left: 95%; animation-delay: 3s; }

    @keyframes fall {
        0% { top: -40px; opacity: 0; transform: rotate(0deg); }
        30% { opacity: 1; }
        100% { top: 100px; opacity: 0; transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

# ========== BERANDA ========== #
if menu == "🏠 Beranda":
    st.title("💬 Deteksi Emosi Teks Bahasa Indonesia")
    st.markdown("""
        Aplikasi ini menggunakan **AI IndoBERT** untuk mendeteksi emosi dari kalimat berbahasa Indonesia.

        ### 🎯 Fitur:
        - Deteksi emosi satu kalimat atau banyak
        - Visualisasi grafik pie chart
        - Ekspor hasil ke CSV
        - Konsultasi Live dengan tim
    """)
    emoji_rain_animation()

# ========== DETEKSI EMOSI ========== #
elif menu == "🧠 Deteksi Emosi":
    st.title("🔍 Deteksi Emosi dari Kalimat")
    emoji_rain_animation()

    contoh_kalimat = st.selectbox("📋 Pilih Contoh Kalimat", (
        "", "Aku senang banget hari ini dapet kabar baik!",
        "Gue marah banget ama dia!",
        "Sedih banget rasanya ditinggal pas sayang-sayangnya.",
        "Stress banget mau UTS tapi belum belajar.",
        "Biasa aja sih, ngga terlalu penting."
    ))

    if contoh_kalimat:
        st.session_state["isi_otomatis"] = contoh_kalimat

    user_input = st.text_area("✍️ Masukkan Teks Kamu", value=st.session_state.get("isi_otomatis", ""))

    if st.button("🚀 Deteksi Sekarang"):
        if user_input.strip():
            with st.spinner("🔍 Mendeteksi emosi... ✨😊😢😠😄"):
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

            st.markdown("#### 💬 Kutipan untuk Kamu:")
            st.info(get_emotion_message(label))

            st.markdown("---")
            st.subheader("🔗 Bagikan Hasil Deteksimu!")
            st.markdown("""
            Ingin temanmu tahu bagaimana suasana hatimu hari ini? Salin teks di bawah dan bagikan ke media sosialmu! 🎉
            """)

            share_text = f"""💬 *Saya baru saja mendeteksi emosi saya lewat AI IndoBERT!*
Teks: "{user_input}"
Emosi: **{label}**

Coba juga deteksi emosi kamu di sini 👉 https://faishal26-emotion-app.streamlit.app"""
            st.code(share_text, language="markdown")
            st.caption("Salin dan bagikan ke WhatsApp, Instagram Story, atau Twitter 🚀")

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
    emoji_rain_animation()
    teks_massal = st.text_area("📝 Masukkan beberapa kalimat (1 baris 1 kalimat):", height=200, placeholder="Contoh:\nAku senang dapet nilai bagus!\nAku kesel banget ama dia!")

    if st.button("🚀 Jalankan Deteksi Massal"):
        if teks_massal.strip():
            kalimat_list = teks_massal.strip().splitlines()
            hasil_massal = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, kalimat in enumerate(kalimat_list):
                emosi, _ = predict_emotion(kalimat)
                hasil_massal.append({"Teks": kalimat, "Emosi": emosi})
                progress = (i + 1) / len(kalimat_list)
                progress_bar.progress(progress)
                status_text.text(f"Mendeteksi {i+1}/{len(kalimat_list)}...")

            status_text.success("Deteksi massal selesai!")
            df_massal = pd.DataFrame(hasil_massal)
            st.dataframe(df_massal)
            st.download_button("💾 Unduh Hasil", data=df_massal.to_csv(index=False), file_name="hasil_massal.csv", mime="text/csv")
        else:
            st.warning("Masukkan setidaknya satu kalimat.")

# ========== KONSULTASI LIVE (BARU) ========== #
elif menu == "💬 Konsultasi Live":
    st.title("💬 Form Konsultasi Live")
    st.markdown("Butuh teman bicara atau saran lebih lanjut? Gunakan form di bawah ini untuk memulai percakapan langsung dengan tim kami.")

    # Link ke Tawk.to chat
    tawk_link = "https://tawk.to/chat/6850d8536134f7190de07c61/1ittsq232"  # Ganti dengan link Anda

    # Menyematkan iframe untuk Tawk.to chat
    st.markdown(f"""
    <iframe src="{tawk_link}" width="100%" height="600" frameborder="0"></iframe>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("❓ Pertanyaan Umum (FAQ)")
    st.markdown("""
    - **Apakah layanan ini gratis?** Ya, konsultasi dasar melalui live chat ini tidak dipungut biaya.
    - **Apakah privasi saya terjamin?** Kami menghargai privasi Anda. Percakapan bersifat rahasia.
    - **Kapan saja saya bisa menggunakan layanan ini?** Tim kami tersedia pada jam kerja (09:00 - 17:00 WIB). Di luar jam tersebut, Anda dapat meninggalkan pesan.
    """)

# ========== TENTANG ========== #
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi Ini")
    emoji_rain_animation()
    st.markdown("""
    Aplikasi deteksi emosi ini dibuat dengan:

    - 🤗 Transformers (IndoBERT)
    - 🔥 PyTorch
    - 🖥️ Streamlit modern style
    - 👨‍💻 Tawk.to

    Dibuat untuk tugas akhir kuliah **Kecerdasan Buatan**.

    👨‍💻 Developer: **Kelompok 1**
    """)
    
    st.caption("© 2025 | Sistem Deteksi Emosi Bahasa Indonesia")
