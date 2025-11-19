import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder

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
st.subheader("Data Berita Terkini")
sector_label = st.selectbox("Pilih Kategori Lapangan Usaha:", options=data['sector_label'].dropna().unique())
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
st.markdown("<br><br>", unsafe_allow_html=True)  # Jeda antara berita dan hasil klasifikasi

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
st.markdown("#### 17 Kategori Lapangan Usaha PDB Dengan Model IndoRoBERTa") 

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
st.markdown("#### Jenis Growth Rate PDB Dengan Model IndoRoBERTa") 

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
    colored_metric2("Akurasi", "89,71%", "#FF9800")
with col2:
    colored_metric2("Presisi", "88,78%", "#4CAF50")
with col3:
    colored_metric2("Recall", "88,64%", "#FFD700")
with col4:
    colored_metric2("F1-Score", "88,71%", "#2196F3")




