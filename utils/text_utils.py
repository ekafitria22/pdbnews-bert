import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Pembersihan umum untuk text pendek / text_for_processing.
    """
    if pd.isnull(text):
        return ""

    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_news_content(text: str) -> str:
    """
    Pembersihan khusus isi berita hasil scraping.
    """
    if pd.isnull(text):
        return ""

    text = str(text)

    text = re.sub(r'Baca juga:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Editor:.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Simak video.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'SCROLL TO CONTINUE WITH CONTENT', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Lihat Video', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Simak Juga Video', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Lihat Juga Video', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\b[A-Z][a-z]+,\s*\d{1,2}\s*[A-Z][a-z]+\s*\d{4},?', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_text_basic(text: str) -> str:
    """
    Pembersihan lanjutan untuk content:
    lowercase, hapus URL, angka, dan karakter non-huruf.
    """
    if pd.isnull(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str):
    if pd.isnull(text):
        return []

    text = str(text).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]
