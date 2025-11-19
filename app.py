import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

st.image("bps.png", width=50, use_column_width=True)

# Judul rata tengah
st.markdown(
    "<h1 style='text-align: center;'>Dashboard Klasifikasi Berita Ekonomi Pergerakan PDB Indonesia</h1>",
    unsafe_allow_html=True
)

# Deskripsi rata tengah
st.markdown(
    "<p style='text-align: center;'>Sistem ini mengklasifikasikan berita ekonomi untuk mendeteksi pergerakan PDB Indonesia. Pengguna dapat melihat hasil klasifikasi pergerakan PDB berdasarkan sektor industri atau lapangan usaha.</p>",
    unsafe_allow_html=True
)

# Load data
data = pd.read_csv("dataset.csv")
data['pdb_label'] = data['pdb_label'].map({1: 'Naik', -1: 'Turun'}).fillna('Tidak diketahui')

# Pilih kategori lapangan usaha
# st.subheader("Data Berita Terkini")
# sector_label = st.selectbox("Pilih Kategori Lapangan Usaha:", options=data['sector_label'].dropna().unique())
# filtered_data = data[data['sector_label'] == sector_label].copy()
# st.write(f"Menampilkan berita dengan kategori: {sector_label}")

# Pilih kategori lapangan usaha
st.subheader("Data Berita Terkini")
sector_label = st.selectbox("Pilih Kategori Lapangan Usaha:", options=["Semua"] + list(data['sector_label'].dropna().unique()))

# Filter data berdasarkan kategori yang dipilih
if sector_label == "Semua":
    filtered_data = data.copy()  # Tampilkan semua data jika "Semua" dipilih
else:
    filtered_data = data[data['sector_label'] == sector_label].copy()

st.write(f"Menampilkan berita dengan kategori: {sector_label}")


# Fungsi untuk label dengan warna
def label_with_color(x):
    if x == 'Naik':
        return "🟢 Naik"
    elif x == 'Turun':
        return "🔴 Turun"
    else:
        return x

filtered_data['pdb_label_color'] = filtered_data['pdb_label'].apply(label_with_color)

# Menampilkan data dengan AgGrid
cols_to_show = ['title', 'publish_date', 'sector_label', 'pdb_label_color', 'growth_label']
data1 = filtered_data[cols_to_show].copy()
data1.reset_index(drop=True, inplace=True)

# AgGrid setup
gb = GridOptionsBuilder.from_dataframe(data1)
gb.configure_default_column(editable=False, groupable=False)
gb.configure_column("title", width=15, header_name="Judul Berita")
gb.configure_column("publish_date", width=50, header_name="Tanggal Terbit")
gb.configure_column("sector_label", width=100, header_name="Sektor Industri")
gb.configure_column("pdb_label_color", width=50, header_name="Prediksi")
gb.configure_column("growth_label", width=50, header_name="Jenis Berita Pertumbuhan")

grid_options = gb.build()

AgGrid(data1, gridOptions=grid_options, height=400)

# Menambahkan jeda antara dua bagian
st.markdown("<br>", unsafe_allow_html=True)  # Jeda antara berita dan hasil klasifikasi

# Hasil klasifikasi pergerakan PDB
st.subheader("Hasil Klasifikasi")
st.markdown("#### Pergerakan PDB Dengan Model IndoRoBERTa") 

# Fungsi untuk membuat teks berwarna
def colored_metric(label, value, color):
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white; text-align: center;">
        <h4>{label}</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

# Menampilkan hasil klasifikasi
col1, col2, col3, col4 = st.columns(4)
with col1:
    colored_metric("Akurasi", "89,71%", "#FF9800")
with col2:
    colored_metric("Presisi", "88,78%", "#4CAF50")
with col3:
    colored_metric("Recall", "88,64%", "#FFD700")
with col4:
    colored_metric("F1-Score", "88,71%", "#2196F3")

# Menambahkan jeda lagi setelah pergerakan PDB
# st.markdown("<br><br>", unsafe_allow_html=True)  # Jeda antara pergerakan PDB dan kategori usaha

# Hasil klasifikasi 17 kategori lapangan usaha
st.markdown("#### 17 Kategori Lapangan Usaha Dengan Model IndoRoBERTa") 

def colored_metric1(label, value, color):
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white; text-align: center;">
        <h4>{label}</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

# Menampilkan hasil klasifikasi
col1, col2, col3, col4 = st.columns(4)
# Menampilkan hasil klasifikasi untuk kategori usaha
with col1:
    colored_metric1("Akurasi", "85,88%", "#FF9800")
with col2:
    colored_metric1("Presisi", "86,21%", "#4CAF50")
with col3:
    colored_metric1("Recall", "86,48%", "#FFD700")
with col4:
    colored_metric1("F1-Score", "86,17%", "#2196F3")

# Hasil klasifikasi 17 kategori lapangan usaha
st.markdown("#### Jenis Berita Pertumbuhan Dengan Model IndoRoBERTa") 

def colored_metric2(label, value, color):
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white; text-align: center;">
        <h4>{label}</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)

# Menampilkan hasil klasifikasi
col1, col2, col3, col4 = st.columns(4)
# Menampilkan hasil klasifikasi untuk kategori usaha
with col1:
    colored_metric2("Akurasi", "82,71%", "#FF9800")
with col2:
    colored_metric2("Presisi", "75,53%", "#4CAF50")
with col3:
    colored_metric2("Recall", "61,25%", "#FFD700")
with col4:
    colored_metric2("F1-Score", "64,88%", "#2196F3")

# Tampilan untuk memilih jenis berita
st.subheader("Pilih Jenis Berita")
news_type = st.selectbox("Pilih Jenis Berita:", options=["Berita Ekonomi", "Berita Politik", "Berita Teknologi", "Berita Olahraga", "Berita Hiburan", "Semua"])

# Tampilan untuk memilih rentang tanggal
st.subheader("Pilih Rentang Tanggal")
start_date = st.date_input("Tanggal Mulai")
end_date = st.date_input("Tanggal Akhir")

# Tombol untuk menyimpan pilihan
st.subheader("Proses Scraping dan Penerapan Model")
save_button = st.button("Simpan Pilihan")

# Tombol untuk memulai proses pembersihan berita
process_button = st.button("Proses Pembersihan Berita")

# Tombol untuk memulai segmentasi berita
segment_button = st.button("Segmentasi Berita")

# Tombol untuk menerapkan model pada berita
apply_model_button = st.button("Terapkan Model")

# Informasi tambahan atau feedback kepada user
if save_button:
    st.write("Pilihan berita dan tanggal telah disimpan.")
if process_button:
    st.write("Proses pembersihan berita dimulai...")
if segment_button:
    st.write("Segmentasi berita sedang diproses...")
if apply_model_button:
    st.write("Model sedang diterapkan pada berita...")












