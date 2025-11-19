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

data = pd.read_csv("dataset.csv")
data['pdb_label'] = data['pdb_label'].map({1: 'Naik', -1: 'Turun'}).fillna('Tidak diketahui')

st.subheader("Data Berita Terkini")
sector_label = st.selectbox("Pilih Kategori Lapangan Usaha:", options=data['sector_label'].dropna().unique())
filtered_data = data[data['sector_label'] == sector_label].copy()
st.write(f"Menampilkan berita dengan kategori: {sector_label}")

def label_with_color(x):
    if x == 'Naik':
        return "🟢 Naik"
    elif x == 'Turun':
        return "🔴 Turun"
    else:
        return x

filtered_data['pdb_label_color'] = filtered_data['pdb_label'].apply(label_with_color)

cols_to_show = ['title', 'publish_date', 'sector_label', 'pdb_label_color']
data1 = filtered_data[cols_to_show].copy()
data1.reset_index(drop=True, inplace=True)

gb = GridOptionsBuilder.from_dataframe(data1)
gb.configure_default_column(editable=False, groupable=False)
gb.configure_column("title", width=300, header_name="Judul Berita")
gb.configure_column("publish_date", width=150, header_name="Tanggal Terbit")
gb.configure_column("sector_label", width=150, header_name="Sektor Industri")
gb.configure_column("pdb_label_color", width=120, header_name="Prediksi")

grid_options = gb.build()

AgGrid(data1, gridOptions=grid_options, height=400)

st.subheader("Hasil Klasifikasi")
st.markdown("#### Dengan Model IndoRoBERTa") 
col1, col2, col3, col4 = st.columns(4)
# Fungsi untuk buat teks berwarna
def colored_metric(label, value, color):
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white; text-align: center;">
        <h4>{label}</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)
with col1:
    colored_metric("Akurasi", "89,71%", "#FF9800")
with col2:
    colored_metric("Presisi", "88,78%", "#4CAF50")
with col3:
    colored_metric("Recall", "88,64%", "#FFD700")
with col4:
    colored_metric("F1-Score", "88,71%", "#2196F3")

def colored_metric(label, value, color):
    st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {color}; color: white; text-align: center;">
        <h4>{label}</h4>
        <p style="font-size: 24px; font-weight: bold; margin: 0;">{value}</p>
    </div>
    """, unsafe_allow_html=True)
with col1:
    colored_metric("Akurasi", "89,71%", "#FF9800")
with col2:
    colored_metric("Presisi", "88,78%", "#4CAF50")
with col3:
    colored_metric("Recall", "88,64%", "#FFD700")
with col4:
    colored_metric("F1-Score", "88,71%", "#2196F3")


