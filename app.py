import os
import ast
import pickle
import csv
from io import BytesIO

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from st_aggrid import AgGrid, GridOptionsBuilder

from utils.constants import APP_TITLE, CATEGORY_SITEID, NEWS_TYPE_OPTIONS, KEYWORD_HINT
from utils.scraper_detik import scrape_detik_search
from utils.text_utils import clean_text, split_sentences

# =========================
# MODEL PATHS
# =========================
CATEGORY_MODEL_DIR = "./category"
MOVEMENT_MODEL_DIR = "./movement"
GROWTH_MODEL_DIR = "./growth"

CATEGORY_ENCODER = "label_encoder_category.pkl"
MOVEMENT_ENCODER = "label_encoder_movement.pkl"
GROWTH_ENCODER = "label_encoder_growth.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CONFIG
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.markdown("""
<style>
div.stButton > button {
    border-radius: 8px;
    padding: 10px 16px;
    border: 0px;
    font-weight: 600;
}
.small-note {font-size: 13px; opacity: 0.8;}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
if os.path.exists("bps.png"):
    st.image("bps.png", width=200)

st.markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>"
    "Aplikasi mendukung dua mode: "
    "<b>(1)</b> scraping berita baru lalu processing dan klasifikasi, "
    "<b>(2)</b> load dataset CSV yang sudah disimpan."
    "</p>",
    unsafe_allow_html=True
)

# =========================
# SESSION STATE INIT
# =========================
for key, default in {
    "params": {},
    "df_raw": pd.DataFrame(),
    "df_clean": pd.DataFrame(),
    "df_pred": pd.DataFrame(),
    "segments": {},
    "loaded_from": "",
    "avg_confidence": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =========================
# MODEL LOADER
# =========================
@st.cache_resource
def load_model_bundle(model_dir, encoder_filename):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()

    encoder_path = os.path.join(model_dir, encoder_filename)
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return tokenizer, model, label_encoder


@st.cache_resource
def load_all_models():
    category_bundle = load_model_bundle(CATEGORY_MODEL_DIR, CATEGORY_ENCODER)
    movement_bundle = load_model_bundle(MOVEMENT_MODEL_DIR, MOVEMENT_ENCODER)
    growth_bundle = load_model_bundle(GROWTH_MODEL_DIR, GROWTH_ENCODER)
    return category_bundle, movement_bundle, growth_bundle


def models_ready():
    checks = [
        os.path.exists(os.path.join(CATEGORY_MODEL_DIR, "config.json")),
        os.path.exists(os.path.join(CATEGORY_MODEL_DIR, CATEGORY_ENCODER)),
        os.path.exists(os.path.join(MOVEMENT_MODEL_DIR, "config.json")),
        os.path.exists(os.path.join(MOVEMENT_MODEL_DIR, MOVEMENT_ENCODER)),
        os.path.exists(os.path.join(GROWTH_MODEL_DIR, "config.json")),
        os.path.exists(os.path.join(GROWTH_MODEL_DIR, GROWTH_ENCODER)),
    ]
    return all(checks)


def predict_single_text(text, tokenizer, model, encoder, max_length=512):
    text = str(text).strip()
    if not text:
        return "", 0.0

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()

    label = encoder.inverse_transform([pred_idx])[0]
    return label, confidence

# =========================
# HELPERS
# =========================
def fmt_ddmmyyyy(d):
    return d.strftime("%d/%m/%Y")


def reset_downstream_state():
    st.session_state.df_clean = pd.DataFrame()
    st.session_state.df_pred = pd.DataFrame()
    st.session_state.segments = {}
    st.session_state.avg_confidence = {}


def normalize_df(df: pd.DataFrame, source_name: str = "dataset.csv") -> pd.DataFrame:
    df = df.copy()

    default_cols = {
        "title": "",
        "category": "",
        "publish_date": "",
        "article_url": "",
        "content": "",
        "segment": "",
        "neural_sentences": "",
        "sector_label": "",
        "pdb_label": "",
        "growth_label": "",
        "source": source_name,
    }

    for col, val in default_cols.items():
        if col not in df.columns:
            if col == "article_url":
                df[col] = [f"row_{i}" for i in range(len(df))]
            elif col == "source":
                df[col] = source_name
            else:
                df[col] = val

    df["source"] = df["source"].fillna(source_name)
    df["article_url"] = df["article_url"].astype(str)
    df["title"] = df["title"].astype(str)
    return df


def robust_read_csv(path_or_buffer):
    attempts = [
        {"sep": ",", "engine": "python", "quoting": csv.QUOTE_MINIMAL},
        {"sep": ",", "engine": "python", "on_bad_lines": "skip"},
        {"sep": None, "engine": "python", "on_bad_lines": "skip"},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            return pd.read_csv(path_or_buffer, **kwargs)
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Gagal membaca CSV. Error terakhir: {last_error}")


def parse_list_string(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
        return [str(parsed).strip()]
    except Exception:
        return [text]


def choose_text_for_processing(row):
    neural_list = parse_list_string(row.get("neural_sentences", ""))
    if neural_list:
        return " ".join(neural_list), neural_list, "neural_sentences"

    content = str(row.get("content", "")).strip()
    if content:
        return content, split_sentences(content), "content"

    title = str(row.get("title", "")).strip()
    return title, split_sentences(title), "title"


def normalize_pdb_label(value):
    s = str(value).strip().lower()
    if s in {"1", "naik"}:
        return "Naik"
    if s in {"0", "-1", "turun"}:
        return "Turun"
    return str(value)


def add_sector_emoji(value):
    s = str(value).strip().lower()

    mapping = {
        "pertanian": "🌾 Pertanian",
        "pertambangan": "⛏️ Pertambangan",
        "industri": "🏭 Industri",
        "listrik": "⚡ Listrik",
        "air": "💧 Air",
        "konstruksi": "🏗️ Konstruksi",
        "perdagangan": "🛒 Perdagangan",
        "transportasi": "🚚 Transportasi",
        "akomodasi": "🏨 Akomodasi",
        "informasi": "💻 Informasi",
        "keuangan": "💰 Keuangan",
        "real_estate": "🏠 Real Estate",
        "jasa_profesional": "🧑‍💼 Jasa Profesional",
        "pemerintahan": "🏛️ Pemerintahan",
        "pendidikan": "🎓 Pendidikan",
        "kesehatan": "🚑 Kesehatan",
        "jasa_lainnya": "🧩 Jasa Lainnya",
    }

    return mapping.get(s, str(value))

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Mode Sumber Data")
    data_source_mode = st.radio(
        "Pilih mode:",
        options=["Scraping Live", "Load CSV Tersimpan"],
        index=1,
    )

    st.markdown("---")
    if models_ready():
        st.success("3 model terdeteksi.")
    else:
        st.warning("Model belum lengkap. Cek folder category, movement, growth.")

    st.caption("Hasil tidak disimpan permanen di server. Gunakan tombol download untuk menyimpan ke komputer Anda.")

# =========================
# INPUT AREA
# =========================
if data_source_mode == "Scraping Live":
    st.subheader("Parameter Scraping")

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Tanggal Mulai")
    with c2:
        end_date = st.date_input("Tanggal Akhir")

    news_type = st.selectbox("Jenis Berita", options=NEWS_TYPE_OPTIONS)
    keywords = st.text_input("Kata kunci berita", value="pdb")

    show_notes = st.checkbox("Tampilkan contoh keyword")
    if show_notes:
        st.write(", ".join(KEYWORD_HINT))

    cat_label = st.selectbox("Kategori Detik", options=["Semua"] + list(CATEGORY_SITEID.keys()))
    max_articles = st.slider("Maksimal artikel", 5, 200, 30, 5)

    with st.expander("Pengaturan request (advanced)"):
        sleep_s = st.slider("Delay per request (detik)", 0.3, 3.0, 1.2, 0.1)
        timeout = st.slider("Timeout (detik)", 5, 60, 30, 5)

else:
    st.subheader("Load Dataset CSV")
    csv_mode = st.radio(
        "Sumber CSV:",
        options=["Gunakan file lokal dataset.csv", "Gunakan file lokal dataset_master.csv", "Upload file CSV"],
        horizontal=True
    )
    uploaded_csv = None
    if csv_mode == "Upload file CSV":
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

st.markdown("<hr>", unsafe_allow_html=True)

# =========================
# BUTTONS
# =========================
if data_source_mode == "Scraping Live":
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        save_clicked = st.button("Simpan Pilihan")
    with b2:
        scrape_clicked = st.button("Proses Scraping")
    with b3:
        segment_clicked = st.button("Processing")
    with b4:
        model_clicked = st.button("Klasifikasikan")
    load_csv_clicked = False
else:
    b1, b2, b3 = st.columns(3)
    with b1:
        load_csv_clicked = st.button("Load CSV")
    with b2:
        segment_clicked = st.button("Processing")
    with b3:
        model_clicked = st.button("Klasifikasikan")
    save_clicked = False
    scrape_clicked = False

# =========================
# ACTIONS
# =========================
if save_clicked and data_source_mode == "Scraping Live":
    st.session_state.params = {
        "start_date": start_date,
        "end_date": end_date,
        "news_type": news_type,
        "keywords": keywords.strip(),
        "cat_label": cat_label,
        "max_articles": int(max_articles),
        "sleep_s": float(sleep_s),
        "timeout": int(timeout),
    }
    st.success("Pilihan tersimpan.")

if load_csv_clicked and data_source_mode == "Load CSV Tersimpan":
    try:
        if csv_mode == "Gunakan file lokal dataset.csv":
            path = "dataset.csv"
            if not os.path.exists(path):
                st.error("File dataset.csv tidak ditemukan.")
                st.stop()
            df_raw = robust_read_csv(path)
            loaded_from = path
        elif csv_mode == "Gunakan file lokal dataset_master.csv":
            path = "dataset_master.csv"
            if not os.path.exists(path):
                st.error("File dataset_master.csv tidak ditemukan.")
                st.stop()
            df_raw = robust_read_csv(path)
            loaded_from = path
        else:
            if uploaded_csv is None:
                st.error("Silakan upload CSV terlebih dahulu.")
                st.stop()
            df_raw = robust_read_csv(uploaded_csv)
            loaded_from = uploaded_csv.name

        df_raw = normalize_df(df_raw, source_name=loaded_from)
        st.session_state.df_raw = df_raw
        st.session_state.loaded_from = loaded_from
        reset_downstream_state()
        st.success(f"Dataset berhasil dimuat: {loaded_from} ({len(df_raw)} baris)")
        st.info(f"Kolom terdeteksi: {', '.join(df_raw.columns.tolist())}")
    except Exception as e:
        st.error(f"Gagal load CSV: {e}")

if scrape_clicked and data_source_mode == "Scraping Live":
    if not keywords.strip():
        st.error("Keyword tidak boleh kosong.")
        st.stop()
    if end_date < start_date:
        st.error("Tanggal akhir harus >= tanggal mulai.")
        st.stop()

    progress = st.progress(0)
    status = st.empty()
    scrape_errors = []
    from_date = fmt_ddmmyyyy(start_date)
    to_date = fmt_ddmmyyyy(end_date)

    def cb(done, total):
        pct = int((done / total) * 100) if total else 0
        progress.progress(min(pct, 100))
        status.write(f"Mengambil {done}/{total} artikel...")

    try:
        df_raw = pd.DataFrame()

        if cat_label == "Semua":
            dfs = []
            total_cat = len(CATEGORY_SITEID)
            for i, (name, siteid) in enumerate(CATEGORY_SITEID.items(), start=1):
                status.write(f"Scraping kategori {i}/{total_cat}: {name}")
                try:
                    df = scrape_detik_search(
                        query=keywords.strip(),
                        siteid=siteid,
                        from_date=from_date,
                        to_date=to_date,
                        max_articles=max_articles,
                        timeout=timeout,
                        sleep_s=sleep_s,
                        progress_cb=None,
                    )
                    if df is not None and not df.empty:
                        df["source"] = name
                        dfs.append(df)
                except Exception as e:
                    scrape_errors.append(f"{name}: {e}")
                progress.progress(int((i / total_cat) * 100))
            df_raw = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        else:
            siteid = CATEGORY_SITEID[cat_label]
            try:
                df_raw = scrape_detik_search(
                    query=keywords.strip(),
                    siteid=siteid,
                    from_date=from_date,
                    to_date=to_date,
                    max_articles=max_articles,
                    timeout=timeout,
                    sleep_s=sleep_s,
                    progress_cb=cb,
                )
                if df_raw is not None and not df_raw.empty:
                    df_raw["source"] = cat_label
                else:
                    df_raw = pd.DataFrame()
            except Exception as e:
                scrape_errors.append(f"{cat_label}: {e}")
                df_raw = pd.DataFrame()

        if not df_raw.empty:
            df_raw = normalize_df(df_raw, source_name="scraping_live")
            if "article_url" in df_raw.columns:
                df_raw = df_raw.drop_duplicates(subset=["article_url"]).reset_index(drop=True)

        st.session_state.df_raw = df_raw
        st.session_state.loaded_from = "scraping_live"
        reset_downstream_state()

        progress.progress(100)
        if scrape_errors:
            st.warning("Sebagian scraping gagal:\n\n" + "\n".join([str(x) for x in scrape_errors[:10]]))

        if df_raw.empty:
            st.warning("Tidak ada artikel ditemukan.")
        else:
            st.success(f"Selesai scraping. Total artikel: {len(df_raw)}")
    except Exception as e:
        st.error(f"Terjadi error saat scraping: {e}")

if segment_clicked:
    if st.session_state.df_raw is None or st.session_state.df_raw.empty:
        st.warning("Belum ada data. Lakukan scraping atau load CSV terlebih dahulu.")
        st.stop()

    df = st.session_state.df_raw.copy()
    text_for_processing = []
    text_clean = []
    segment_source = []
    seg_map = {}

    for idx, row in df.iterrows():
        article_id = row.get("article_url", f"row_{idx}")
        chosen_text, chosen_segments, source_name = choose_text_for_processing(row)

        text_for_processing.append(chosen_text)
        text_clean.append(clean_text(chosen_text))
        segment_source.append(source_name)
        seg_map[article_id] = chosen_segments

    df["text_for_processing"] = text_for_processing
    df["text_clean"] = text_clean
    df["segment_source"] = segment_source

    st.session_state.df_clean = df
    st.session_state.segments = seg_map

    msg = "Processing selesai menggunakan "
    if (df["segment_source"] == "neural_sentences").any():
        msg += "neural_sentences."
    elif (df["segment_source"] == "content").any():
        msg += "content."
    else:
        msg += "title."
    st.success(msg)

if model_clicked:
    if st.session_state.df_clean is None or st.session_state.df_clean.empty:
        st.warning("Belum ada data hasil processing.")
        st.stop()

    if not models_ready():
        st.error("Folder model/encoder belum lengkap. Cek category, movement, dan growth.")
        st.stop()

    try:
        (category_tokenizer, category_model, category_encoder), \
        (movement_tokenizer, movement_model, movement_encoder), \
        (growth_tokenizer, growth_model, growth_encoder) = load_all_models()

        df = st.session_state.df_clean.copy()

        progress = st.progress(0)
        total = len(df)

        sector_preds = []
        sector_confs = []
        movement_preds = []
        movement_confs = []
        growth_preds = []
        growth_confs = []

        for i, text in enumerate(df["text_for_processing"].astype(str).tolist(), start=1):
            sector_label, sector_conf = predict_single_text(text, category_tokenizer, category_model, category_encoder)
            movement_label, movement_conf = predict_single_text(text, movement_tokenizer, movement_model, movement_encoder)
            growth_label, growth_conf = predict_single_text(text, growth_tokenizer, growth_model, growth_encoder)

            sector_preds.append(sector_label)
            sector_confs.append(sector_conf)
            movement_preds.append(movement_label)
            movement_confs.append(movement_conf)
            growth_preds.append(growth_label)
            growth_confs.append(growth_conf)

            progress.progress(int((i / total) * 100))

        df["sector_label"] = sector_preds
        df["sector_confidence"] = sector_confs
        df["pdb_label"] = movement_preds
        df["pdb_confidence"] = movement_confs
        df["growth_label"] = growth_preds
        df["growth_confidence"] = growth_confs

        st.session_state.df_pred = df
        st.session_state.avg_confidence = {
            "sector": float(pd.Series(sector_confs).mean()) if sector_confs else 0.0,
            "pdb": float(pd.Series(movement_confs).mean()) if movement_confs else 0.0,
            "growth": float(pd.Series(growth_confs).mean()) if growth_confs else 0.0,
        }

        st.success("Klasifikasi model selesai.")
    except Exception as e:
        st.error(f"Gagal menjalankan model: {e}")

# =========================
# DISPLAY
# =========================
st.subheader("Hasil Berita")

df_show = st.session_state.df_pred if not st.session_state.df_pred.empty else st.session_state.df_raw

if df_show is None or df_show.empty:
    st.info("Belum ada data.")
else:
    filtered = df_show.copy()

    if "pdb_label" in filtered.columns:
        filtered["pdb_label_display"] = filtered["pdb_label"].apply(normalize_pdb_label)
    else:
        filtered["pdb_label_display"] = ""

    if "sector_label" in filtered.columns:
        filtered["sector_label_emoji"] = filtered["sector_label"].apply(add_sector_emoji)
    else:
        filtered["sector_label_emoji"] = ""

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        sector_options = ["Semua"]
        if "sector_label" in filtered.columns:
            vals = [str(x) for x in filtered["sector_label"].dropna().unique().tolist() if str(x).strip()]
            sector_options += sorted(vals)
        sector_filter = st.selectbox("Filter sektor", sector_options)

    with filter_col2:
        pdb_options = ["Semua"]
        if "pdb_label_display" in filtered.columns:
            vals = [str(x) for x in filtered["pdb_label_display"].dropna().unique().tolist() if str(x).strip()]
            pdb_options += sorted(vals)
        pdb_filter = st.selectbox("Filter pergerakan PDB", pdb_options)

    with filter_col3:
        growth_options = ["Semua"]
        if "growth_label" in filtered.columns:
            vals = [str(x) for x in filtered["growth_label"].dropna().unique().tolist() if str(x).strip()]
            growth_options += sorted(vals)
        growth_filter = st.selectbox("Filter growth", growth_options)

    if sector_filter != "Semua" and "sector_label" in filtered.columns:
        filtered = filtered[filtered["sector_label"].astype(str) == sector_filter].copy()

    if pdb_filter != "Semua" and "pdb_label_display" in filtered.columns:
        filtered = filtered[filtered["pdb_label_display"].astype(str) == pdb_filter].copy()

    if growth_filter != "Semua" and "growth_label" in filtered.columns:
        filtered = filtered[filtered["growth_label"].astype(str) == growth_filter].copy()

    if "pdb_label_display" in filtered.columns:
        filtered["pdb_label_color"] = filtered["pdb_label_display"].apply(
            lambda x: "🟢 Naik" if str(x).strip().lower() == "naik"
            else ("🔴 Turun" if str(x).strip().lower() == "turun" else str(x))
        )
    else:
        filtered["pdb_label_color"] = ""

    cols_to_show = [
        "title", "publish_date", "category", "source",
        "segment_source", "sector_label_emoji", "sector_confidence",
        "pdb_label_color", "pdb_confidence",
        "growth_label", "growth_confidence"
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]
    view_df = filtered[cols_to_show].reset_index(drop=True)

    gb = GridOptionsBuilder.from_dataframe(view_df)
    gb.configure_default_column(editable=False, groupable=False, resizable=True, sortable=True, filter=True)

    if "title" in view_df.columns:
        gb.configure_column("title", header_name="Judul Berita", width=420)
    if "publish_date" in view_df.columns:
        gb.configure_column("publish_date", header_name="Tanggal Terbit", width=160)
    if "category" in view_df.columns:
        gb.configure_column("category", header_name="Kategori", width=120)
    if "source" in view_df.columns:
        gb.configure_column("source", header_name="Sumber", width=120)
    if "segment_source" in view_df.columns:
        gb.configure_column("segment_source", header_name="Sumber Kalimat", width=140)
    if "sector_label_emoji" in view_df.columns:
        gb.configure_column("sector_label_emoji", header_name="Sektor", width=180)
    if "sector_confidence" in view_df.columns:
        gb.configure_column("sector_confidence", header_name="Conf. Sektor", width=120)
    if "pdb_label_color" in view_df.columns:
        gb.configure_column("pdb_label_color", header_name="Pergerakan PDB", width=140)
    if "pdb_confidence" in view_df.columns:
        gb.configure_column("pdb_confidence", header_name="Conf. PDB", width=110)
    if "growth_label" in view_df.columns:
        gb.configure_column("growth_label", header_name="Growth", width=120)
    if "growth_confidence" in view_df.columns:
        gb.configure_column("growth_confidence", header_name="Conf. Growth", width=130)

    AgGrid(view_df, gridOptions=gb.build(), height=420)

    confidence_cols = ["sector_confidence", "pdb_confidence", "growth_confidence"]
    has_confidence = any(col in filtered.columns for col in confidence_cols)

    if has_confidence and not filtered.empty:
        avg_sector_filtered = filtered["sector_confidence"].mean() if "sector_confidence" in filtered.columns else None
        avg_pdb_filtered = filtered["pdb_confidence"].mean() if "pdb_confidence" in filtered.columns else None
        avg_growth_filtered = filtered["growth_confidence"].mean() if "growth_confidence" in filtered.columns else None

        st.markdown("### Rata-rata Confidence Prediksi")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Category",
                f"{avg_sector_filtered:.2%}" if avg_sector_filtered is not None and pd.notna(avg_sector_filtered) else "-"
            )
        with c2:
            st.metric(
                "Movement",
                f"{avg_pdb_filtered:.2%}" if avg_pdb_filtered is not None and pd.notna(avg_pdb_filtered) else "-"
            )
        with c3:
            st.metric(
                "Growth",
                f"{avg_growth_filtered:.2%}" if avg_growth_filtered is not None and pd.notna(avg_growth_filtered) else "-"
            )

    with st.expander("Preview teks yang dipakai untuk processing"):
        preview_cols = [c for c in ["title", "text_for_processing", "neural_sentences", "content"] if c in filtered.columns]
        if preview_cols:
            st.dataframe(filtered[preview_cols].head(5), use_container_width=True)

    st.markdown("### Download hasil")
    st.caption("Confidence ikut tersimpan di file CSV dan Excel yang didownload.")

    download_col1, download_col2 = st.columns(2)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    with download_col1:
        st.download_button(
            "Download hasil saat ini (CSV)",
            data=csv_bytes,
            file_name="hasil_berita_ekonomi.csv",
            mime="text/csv",
        )

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="hasil_berita")
    excel_buffer.seek(0)

    with download_col2:
        st.download_button(
            "Download hasil saat ini (Excel)",
            data=excel_buffer.getvalue(),
            file_name="hasil_berita_ekonomi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
