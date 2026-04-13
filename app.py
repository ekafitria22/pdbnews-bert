import os
import ast
import pickle
import csv
import re
from io import BytesIO

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download
from st_aggrid import AgGrid, GridOptionsBuilder

from utils.constants import APP_TITLE, CATEGORY_SITEID, KEYWORD_HINT
from utils.scraper_detik import scrape_detik_search
from utils.text_utils import (
    clean_text,
    clean_news_content,
    clean_text_basic,
    split_sentences,
)

CATEGORY_MODEL_DIR = "ekafitria22/IndoRoBERTa-Base-PDBNews-Category"
MOVEMENT_MODEL_DIR = "ekafitria22/IndoRoBERTa-Base-PDBNews-Movement"
GROWTH_MODEL_DIR = "ekafitria22/IndoRoBERTa-Base-PDBNews-Growth"

CATEGORY_ENCODER = "label_encoder_category.pkl"
MOVEMENT_ENCODER = "label_encoder_movement.pkl"
GROWTH_ENCODER = "label_encoder_growth.pkl"

SBERT_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if os.path.exists("bps.png"):
    st.image("bps.png", width=200)

st.markdown(f"<h1 style='text-align: center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>"
    "Aplikasi mendukung dua mode: "
    "<b>(1)</b> scraping berita baru lalu processing, seleksi kalimat, dan klasifikasi, "
    "<b>(2)</b> load dataset CSV yang sudah disimpan."
    "</p>",
    unsafe_allow_html=True
)

for key, default in {
    "params": {},
    "df_raw": pd.DataFrame(),
    "df_clean": pd.DataFrame(),
    "df_selected": pd.DataFrame(),
    "df_pred": pd.DataFrame(),
    "segments": {},
    "loaded_from": "",
    "avg_confidence": {},
    "clean_result_df": pd.DataFrame(),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

@st.cache_resource
def load_model_bundle(model_repo_id: str, encoder_filename: str):
    tokenizer = AutoTokenizer.from_pretrained(model_repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo_id)
    model.to(DEVICE)
    model.eval()

    encoder_path = hf_hub_download(
        repo_id=model_repo_id,
        filename=encoder_filename,
        repo_type="model",
    )
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    return tokenizer, model, label_encoder

@st.cache_resource
def load_all_models():
    category_bundle = load_model_bundle(CATEGORY_MODEL_DIR, CATEGORY_ENCODER)
    movement_bundle = load_model_bundle(MOVEMENT_MODEL_DIR, MOVEMENT_ENCODER)
    growth_bundle = load_model_bundle(GROWTH_MODEL_DIR, GROWTH_ENCODER)
    return category_bundle, movement_bundle, growth_bundle

@st.cache_resource
def load_sbert_model():
    return SentenceTransformer(SBERT_MODEL_NAME)

def models_ready() -> bool:
    try:
        hf_hub_download(repo_id=CATEGORY_MODEL_DIR, filename=CATEGORY_ENCODER, repo_type="model")
        hf_hub_download(repo_id=MOVEMENT_MODEL_DIR, filename=MOVEMENT_ENCODER, repo_type="model")
        hf_hub_download(repo_id=GROWTH_MODEL_DIR, filename=GROWTH_ENCODER, repo_type="model")
        return True
    except Exception:
        return False

def predict_single_text(text: str, tokenizer, model, encoder, max_length: int = 512):
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

SELECTION_KEYWORDS = [
    "naik", "tumbuh", "tingkat", "positif", "optimis", "kuat",
    "lonjak", "puncak", "baik", "pulih", "lebih", "deflasi", "stimulus",
    "bangun ekonomi", "surplus", "capai", "hijau", "peningkatan", "peluang",
    "meningkat", "tumbuh pesat", "optimisme", "akselerasi", "stabil", "kinerja",
    "progres", "proyeksi", "pertumbuhan ekonomi", "lonjakan sektor", "peningkatan yang signifikan",
    "peluang pertumbuhan", "akselerasi pertumbuhan", "stabilisasi ekonomi", "proyeksi positif",
    "ekspansi ekonomi", "kinerja yang kuat", "kekuatan ekonomi", "ekonomi yang pulih",
    "peningkatan produktivitas", "pertumbuhan investasi", "peluang pasar", "pendapatan meningkat",
    "optimisme pasar", "optimisme investor", "keuntungan besar", "pencapaian ekonomi",
    "peningkatan daya beli", "peningkatan lapangan pekerjaan", "proyeksi pertumbuhan stabil",
    "indikator ekonomi positif", "sektor berkembang", "sektor ekspansif", "kondisi ekonomi sehat",
    "kepercayaan ekonomi", "kinerja sektor", "investasi berkembang", "kinerja ekspor",
    "inflasi", "turun", "lambat", "jatuh", "lemah", "merosot",
    "kontraksi", "rebound", "puruk", "buruk", "susut", "krisis", "resesi",
    "negatif", "gagal", "jatuh bebas", "anggur", "defisit",
    "miskin", "efisiensi", "drop", "merah", "penurunan",
    "deflasi", "penyusutan", "tertahan", "lesu", "depresi", "kemerosotan", "berkurang",
    "penurunan ekonomi", "deflasi ekonomi", "kemerosotan sektor", "krisis ekonomi",
    "resesi global", "kontraksi ekonomi", "lambatnya pertumbuhan", "penyusutan sektor",
    "kondisi lesu", "pertumbuhan negatif", "berkurangnya investasi", "krisis finansial",
    "penurunan daya beli", "krisis moneter", "resesi ekonomi global", "penurunan pendapatan",
    "krisis utang", "kondisi pasar buruk", "turunnya konsumsi", "kelemahan sektor",
    "penurunan produksi", "depresi ekonomi", "kemerosotan pasar", "penyusutan pasar",
    "beban utang tinggi", "kelemahan investasi", "turunnya ekspor", "krisis ketenagakerjaan",
    "pengangguran meningkat", "defisit fiskal",
    "tahun", "perbandingan tahunan", "dari tahun ke tahun", "pertumbuhan tahunan",
    "perbandingan tahun sebelumnya", "pertumbuhan ekonomi tahun ini", "analisis tahunan",
    "yoy", "y o y", "y-o-y", "year-on-year", "year on year",
    "kuartal", "triwulan", "perbandingan kuartalan", "pertumbuhan kuartalan",
    "qtoq", "q-to-q", "quartal-to-quartal", "perbandingan kuartal sebelumnya",
    "pertumbuhan ekonomi kuartal ini", "analisis kuartalan", "quartal to quartal",
    "q to q", "q1", "q2", "q3", "q4",
    "pertumbuhan kumulatif", "c-to-c", "cumulative", "tahun berjalan",
    "semester pertama", "semester kedua", "setengah tahun", "kumulatif", "tahun penuh",
    "ctoc", "c to c", "cumulative on cumulative"
]

def fmt_ddmmyyyy(d):
    return d.strftime("%d/%m/%Y")

def reset_downstream_state():
    st.session_state.df_clean = pd.DataFrame()
    st.session_state.df_selected = pd.DataFrame()
    st.session_state.df_pred = pd.DataFrame()
    st.session_state.segments = {}
    st.session_state.avg_confidence = {}

def normalize_df(df: pd.DataFrame, source_name: str = "dataset.csv") -> pd.DataFrame:
    df = df.copy()

    if "row_id" not in df.columns:
        df["row_id"] = [f"{source_name}_{i}" for i in range(len(df))]

    default_cols = {
        "title": "",
        "category": "",
        "publish_date": "",
        "article_url": "",
        "content": "",
        "cleaned_content": "",
        "segment": "",
        "segmented_content": "",
        "neural_sentences": "",
        "selected_sentences": "",
        "selected_text": "",
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
    df["row_id"] = df["row_id"].astype(str)
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
    selected_list = parse_list_string(row.get("selected_sentences", ""))
    if selected_list:
        return " ".join(selected_list), selected_list, "selected_sentences"

    neural_list = parse_list_string(row.get("neural_sentences", ""))
    if neural_list:
        return " ".join(neural_list), neural_list, "neural_sentences"

    segmented_content = row.get("segmented_content", [])
    if isinstance(segmented_content, list) and segmented_content:
        return " ".join(segmented_content), segmented_content, "segmented_content"

    cleaned_content = str(row.get("cleaned_content", "")).strip()
    if cleaned_content:
        return cleaned_content, split_sentences(cleaned_content), "cleaned_content"

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

def select_sentences_based_on_keywords(sentences, keywords):
    selected = []
    normalized_keywords = [str(k).strip().lower() for k in keywords if str(k).strip()]
    for sentence in sentences:
        s_low = str(sentence).lower()
        if any(keyword in s_low for keyword in normalized_keywords):
            selected.append(sentence)
    return selected

def extract_neural_sentences(sentences, keywords, sbert_model, top_k: int = 5):
    selected_sentences = select_sentences_based_on_keywords(sentences, keywords)
    if not selected_sentences:
        return [sentences[0]] if sentences else []
    if len(selected_sentences) > 100:
        selected_sentences = selected_sentences[:100]

    sentence_embeddings = sbert_model.encode(selected_sentences, batch_size=32, show_progress_bar=False)
    all_text = " ".join(selected_sentences)
    words = re.findall(r"\b\w+\b", all_text.lower())

    if not words:
        return selected_sentences[:top_k]

    unique_words = list(dict.fromkeys(words))
    word_subset = unique_words[:500]
    word_embeddings = sbert_model.encode(word_subset, batch_size=32, show_progress_bar=False)

    cosine_similarities = cosine_similarity(sentence_embeddings, word_embeddings)
    sentence_scores = cosine_similarities.mean(axis=1)
    idx_sorted = sentence_scores.argsort()[::-1]
    top_indices = idx_sorted[:top_k]
    top_sentences = [selected_sentences[i] for i in top_indices]
    return top_sentences

with st.sidebar:
    st.header("Mode Sumber Data")
    data_source_mode = st.radio(
        "Pilih mode:",
        options=["Scraping Berita Real-Time", "Load CSV Berita Tersimpan"],
        index=1,
    )
    st.markdown("---")
    if models_ready():
        st.success("3 model terdeteksi dari Hugging Face.")
    else:
        st.warning("Model atau label encoder dari Hugging Face belum bisa diakses.")
    st.caption("Hasil tidak disimpan permanen di server. Gunakan tombol download untuk menyimpan ke komputer Anda.")

if data_source_mode == "Scraping Berita Real-Time":
    st.subheader("Scraping Berita")

    topic_mode = st.radio(
        "Topik Berita",
        options=["Cari topik berita", "Topik berita campuran"],
        horizontal=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Tanggal Mulai")
    with c2:
        end_date = st.date_input("Tanggal Akhir")

    if topic_mode == "Cari topik berita":
        keywords = st.text_input("Cari topik berita", value="pdb")
        show_notes = st.checkbox("Tampilkan contoh keyword")
        if show_notes:
            st.write(", ".join(KEYWORD_HINT))
    else:
        keywords = "ekonomi"

    cat_label = st.selectbox("Kategori Berita", options=["Semua Kategori"] + list(CATEGORY_SITEID.keys()))
    include_advertorial = st.radio("Sertakan artikel advertorial?", options=["Ya", "Tidak"], horizontal=True, index=1)
    max_articles = st.slider("Jumlah artikel yang ingin diekstrak", 5, 200, 30, 5)

    with st.expander("Pengaturan request (advanced)"):
        sleep_s = st.slider("Delay per request (detik)", 0.5, 3.0, 1.5, 0.1)
        timeout = st.slider("Timeout (detik)", 10, 60, 30, 5)

    exclude_advertorial = include_advertorial == "Tidak"
else:
    st.subheader("Load Dataset CSV")
    csv_mode = st.radio(
        "Sumber CSV:",
        options=["Gunakan file lokal dataset.csv", "Upload file CSV"],
        horizontal=True
    )
    uploaded_csv = None
    if csv_mode == "Upload file CSV":
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

st.markdown("<hr>", unsafe_allow_html=True)

if data_source_mode == "Scraping Berita Real-Time":
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        save_clicked = st.button("Simpan Pilihan")
    with b2:
        scrape_clicked = st.button("Proses Scraping")
    with b3:
        segment_clicked = st.button("Processing")
    with b4:
        select_clicked = st.button("Seleksi Kalimat")
    with b5:
        model_clicked = st.button("Klasifikasikan")
    load_csv_clicked = False
else:
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        load_csv_clicked = st.button("Load CSV")
    with b2:
        segment_clicked = st.button("Processing")
    with b3:
        select_clicked = st.button("Seleksi Kalimat")
    with b4:
        model_clicked = st.button("Klasifikasikan")
    save_clicked = False
    scrape_clicked = False

if save_clicked and data_source_mode == "Scraping Berita Real-Time":
    st.session_state.params = {
        "topic_mode": topic_mode,
        "start_date": start_date,
        "end_date": end_date,
        "keywords": keywords.strip(),
        "cat_label": cat_label,
        "max_articles": int(max_articles),
        "sleep_s": float(sleep_s),
        "timeout": int(timeout),
        "include_advertorial": include_advertorial,
        "exclude_advertorial": bool(exclude_advertorial),
    }
    st.success("Pilihan tersimpan.")

if load_csv_clicked and data_source_mode == "Load CSV Berita Tersimpan":
    try:
        if csv_mode == "Gunakan file lokal dataset.csv":
            path = "dataset.csv"
            if not os.path.exists(path):
                st.error("File dataset.csv tidak ditemukan.")
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

if scrape_clicked and data_source_mode == "Scraping Berita Real-Time":
    if topic_mode == "Cari topik berita" and not keywords.strip():
        st.error("Pencarian topik berita tidak boleh kosong.")
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

        if cat_label == "Semua Kategori":
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
                        include_content=True,
                        exclude_advertorial=exclude_advertorial,
                    )
                    err = getattr(df, "attrs", {}).get("error_message", "")
                    if err:
                        scrape_errors.append(f"{name}: {err}")
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
                    include_content=True,
                    exclude_advertorial=exclude_advertorial,
                )
                err = getattr(df_raw, "attrs", {}).get("error_message", "")
                if err:
                    scrape_errors.append(f"{cat_label}: {err}")
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
            if "row_id" not in df_raw.columns or df_raw["row_id"].isna().any():
                df_raw["row_id"] = [f"scraping_live_{i}" for i in range(len(df_raw))]

        st.session_state.df_raw = df_raw
        st.session_state.loaded_from = "scraping_live"
        reset_downstream_state()

        progress.progress(100)

        if scrape_errors:
            st.warning("Sebagian scraping gagal:\n\n" + "\n".join([str(x) for x in scrape_errors[:10]]))

        if df_raw.empty:
            if scrape_errors:
                st.error("Scraping tidak menghasilkan artikel. Kemungkinan request ke Detik gagal atau halaman tidak berhasil diparse.")
            elif topic_mode == "Topik berita campuran":
                st.warning("Tidak ada artikel ditemukan untuk mode topik berita campuran.")
            else:
                st.warning(f'Tidak ada artikel yang ditemukan mengenai "{keywords}".')
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
    cleaned_contents = []
    segmented_contents = []

    for idx, row in df.iterrows():
        article_id = row.get("row_id") or row.get("article_url", f"row_{idx}")
        raw_content = str(row.get("content", "")).strip()

        cleaned_content = clean_news_content(raw_content)
        segmented_raw = split_sentences(cleaned_content)
        segmented = [clean_text_basic(s) for s in segmented_raw if clean_text_basic(s).strip()]
        cleaned_basic = clean_text_basic(cleaned_content)

        cleaned_contents.append(cleaned_basic)
        segmented_contents.append(segmented)

        chosen_text, chosen_segments, source_name = choose_text_for_processing(row)

        if not chosen_text.strip() and cleaned_content.strip():
            chosen_text = cleaned_content
            chosen_segments = segmented
            source_name = "cleaned_content"

        text_for_processing.append(chosen_text)
        text_clean.append(clean_text(chosen_text))
        segment_source.append(source_name)
        seg_map[article_id] = chosen_segments

    df["cleaned_content"] = cleaned_contents
    df["segmented_content"] = segmented_contents
    df["text_for_processing"] = text_for_processing
    df["text_clean"] = text_clean
    df["segment_source"] = segment_source

    st.session_state.df_clean = df
    st.session_state.segments = seg_map

    msg = "Processing selesai menggunakan "
    if (df["segment_source"] == "selected_sentences").any():
        msg += "selected_sentences."
    elif (df["segment_source"] == "neural_sentences").any():
        msg += "neural_sentences."
    elif (df["segment_source"] == "segmented_content").any():
        msg += "segmented_content."
    elif (df["segment_source"] == "cleaned_content").any():
        msg += "cleaned_content."
    elif (df["segment_source"] == "content").any():
        msg += "content."
    else:
        msg += "title."
    st.success(msg)

if select_clicked:
    source_df = None

    if st.session_state.df_clean is not None and not st.session_state.df_clean.empty:
        source_df = st.session_state.df_clean.copy()
    elif st.session_state.df_raw is not None and not st.session_state.df_raw.empty:
        source_df = st.session_state.df_raw.copy()
    else:
        st.warning("Belum ada data. Lakukan scraping atau load CSV terlebih dahulu.")
        st.stop()

    if "content" not in source_df.columns:
        st.error("Kolom content tidak ditemukan. Seleksi kalimat membutuhkan isi berita pada kolom content.")
        st.stop()

    sbert_model = load_sbert_model()
    selected_sentences_col = []
    selected_text_col = []

    progress = st.progress(0)
    total = len(source_df)

    for i, row in source_df.iterrows():
        existing_segments = row.get("segmented_content", [])

        if isinstance(existing_segments, list) and existing_segments:
            sentences = [str(s).strip() for s in existing_segments if str(s).strip()]
        else:
            raw_content = str(row.get("content", "")).strip()
            cleaned_content = clean_news_content(raw_content)
            segmented_raw = split_sentences(cleaned_content)
            sentences = [clean_text_basic(s) for s in segmented_raw if clean_text_basic(s).strip()]

        selected_sentences = extract_neural_sentences(
            sentences,
            SELECTION_KEYWORDS,
            sbert_model,
            top_k=5
        )

        selected_sentences_col.append(selected_sentences)
        selected_text_col.append(" ".join(selected_sentences))
        progress.progress(int((i + 1) / total * 100))

    source_df["selected_sentences"] = selected_sentences_col
    source_df["selected_text"] = selected_text_col
    source_df["text_for_processing"] = source_df["selected_text"].astype(str)
    source_df["text_clean"] = source_df["text_for_processing"].astype(str).apply(clean_text)
    source_df["segment_source"] = "selected_sentences"

    st.session_state.df_selected = source_df
    st.session_state.df_clean = source_df

    st.success("Seleksi kalimat selesai. Kolom selected_sentences dan selected_text berhasil dibuat.")

if model_clicked:
    if st.session_state.df_clean is None or st.session_state.df_clean.empty:
        st.warning("Belum ada data hasil processing.")
        st.stop()

    if not models_ready():
        st.error("Model dari Hugging Face belum siap diakses.")
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

st.subheader("Hasil Berita")

df_show = st.session_state.df_pred if not st.session_state.df_pred.empty else (
    st.session_state.df_clean if not st.session_state.df_clean.empty else st.session_state.df_raw
)

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
        "row_id",
        "title", "publish_date", "source",
        "segment_source", "sector_label_emoji", "sector_confidence",
        "pdb_label_color", "pdb_confidence",
        "growth_label", "growth_confidence"
    ]
    cols_to_show = [c for c in cols_to_show if c in filtered.columns]
    view_df = filtered[cols_to_show].reset_index(drop=True)

    gb = GridOptionsBuilder.from_dataframe(view_df)
    gb.configure_default_column(
        editable=False,
        groupable=False,
        resizable=True,
        sortable=True,
        filter=True,
    )

    if "row_id" in view_df.columns:
        gb.configure_column("row_id", hide=True)

    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
        header_checkbox=True,
        header_checkbox_filtered_only=False,
    )

    gb.configure_grid_options(
        rowSelection="multiple",
        rowMultiSelectWithClick=True,
        suppressRowClickSelection=False,
    )

    grid_options = gb.build()
    grid_options["rowSelection"] = "multiple"
    grid_options["suppressRowClickSelection"] = False

    grid_response = AgGrid(
        view_df,
        gridOptions=grid_options,
        height=420,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        key="hasil_berita_grid",
    )

    selected_rows = grid_response.get("selected_rows", [])

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        delete_selected_clicked = st.button("Hapus semua baris terpilih")

    with action_col2:
        delete_and_save_clicked = st.button("Hapus terpilih + siapkan hasil bersih")

    if delete_selected_clicked or delete_and_save_clicked:
        if selected_rows is None or len(selected_rows) == 0:
            st.warning("Pilih minimal satu baris terlebih dahulu lewat checkbox di sisi kiri tabel.")
        else:
            selected_df = pd.DataFrame(selected_rows)

            if "row_id" not in selected_df.columns:
                st.error("row_id tidak ditemukan pada baris terpilih.")
            else:
                selected_ids = selected_df["row_id"].astype(str).tolist()

                target_key = "df_pred" if not st.session_state.df_pred.empty else (
                    "df_clean" if not st.session_state.df_clean.empty else "df_raw"
                )

                active_df = st.session_state[target_key].copy()
                before_count = len(active_df)

                active_df = active_df[~active_df["row_id"].astype(str).isin(selected_ids)].copy()
                after_count = len(active_df)

                st.session_state[target_key] = active_df

                if {"sector_confidence", "pdb_confidence", "growth_confidence"}.intersection(active_df.columns):
                    st.session_state.avg_confidence = {
                        "sector": float(active_df["sector_confidence"].mean()) if "sector_confidence" in active_df.columns and not active_df.empty else 0.0,
                        "pdb": float(active_df["pdb_confidence"].mean()) if "pdb_confidence" in active_df.columns and not active_df.empty else 0.0,
                        "growth": float(active_df["growth_confidence"].mean()) if "growth_confidence" in active_df.columns and not active_df.empty else 0.0,
                    }

                if delete_and_save_clicked:
                    try:
                        active_df.to_csv("hasil_bersih_berita.csv", index=False)
                    except Exception:
                        pass

                st.success(f"{before_count - after_count} baris berhasil dihapus.")
                st.rerun()

    confidence_cols = ["sector_confidence", "pdb_confidence", "growth_confidence"]
    has_confidence = any(col in filtered.columns for col in confidence_cols)

    if has_confidence and not filtered.empty:
        avg_sector_filtered = filtered["sector_confidence"].mean() if "sector_confidence" in filtered.columns else None
        avg_pdb_filtered = filtered["pdb_confidence"].mean() if "pdb_confidence" in filtered.columns else None
        avg_growth_filtered = filtered["growth_confidence"].mean() if "growth_confidence" in filtered.columns else None

        st.markdown("### Rata-rata Confidence Prediksi")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Category", f"{avg_sector_filtered:.2%}" if avg_sector_filtered is not None and pd.notna(avg_sector_filtered) else "-")
        with c2:
            st.metric("Movement", f"{avg_pdb_filtered:.2%}" if avg_pdb_filtered is not None and pd.notna(avg_pdb_filtered) else "-")
        with c3:
            st.metric("Growth", f"{avg_growth_filtered:.2%}" if avg_growth_filtered is not None and pd.notna(avg_growth_filtered) else "-")

    with st.expander("Preview teks yang dipakai untuk processing"):
        preview_cols = [
            c for c in [
                "title",
                "content",
                "cleaned_content",
                "segmented_content",
                "selected_sentences",
                "selected_text",
                "text_for_processing",
            ] if c in filtered.columns
        ]
        if preview_cols:
            st.dataframe(filtered[preview_cols].head(5), width="stretch")

    st.markdown("### Download hasil")
    st.caption("Confidence ikut tersimpan di file CSV dan Excel yang didownload.")

    export_df = st.session_state.clean_result_df if not st.session_state.clean_result_df.empty else filtered

    download_col1, download_col2 = st.columns(2)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    with download_col1:
        st.download_button(
            "Download hasil saat ini (CSV)",
            data=csv_bytes,
            file_name="hasil_berita_ekonomi.csv",
            mime="text/csv",
        )

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="hasil_berita")
    excel_buffer.seek(0)

    with download_col2:
        st.download_button(
            "Download hasil saat ini (Excel)",
            data=excel_buffer.getvalue(),
            file_name="hasil_berita_ekonomi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if not st.session_state.clean_result_df.empty:
        st.caption(f"Hasil bersih siap diunduh: {len(st.session_state.clean_result_df)} baris.")
