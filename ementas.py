# analiseementasstreamlit.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import zipfile
import tempfile
import pdfplumber
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap                    # pip install umap-learn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import openai
import xlsxwriter               # para formatação condicional no Excel
from xlsxwriter.utility import xl_rowcol_to_cell

# --------------------------------------------------
# 1) Configuração da página Streamlit
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("📂📑 Análise de Ementas via pasta .zip")

# --------------------------------------------------
# 2) Cache de recursos pesados
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_ementas(zip_bytes: bytes) -> pd.DataFrame:
    """Extrai e limpa ementas de um ZIP de PDFs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        z = zipfile.ZipFile(BytesIO(zip_bytes))
        z.extractall(tmpdir)
        regs = []
        for root, _, files in os.walk(tmpdir):
            for fn in files:
                if not fn.lower().endswith(".pdf"):
                    continue
                path = os.path.join(root, fn)
                txt = ""
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        txt += (p.extract_text() or "") + "\n"
                # remove rodapés tipo "2 de 3"
                txt = re.sub(r"(?m)^\s*\d+\s+de\s+\d+\s*$", "", txt)
                # extrai nome e código
                m = re.search(
                    r"UNIDADE CURRICULAR[:\s]*(.+?)\s*\(\s*(\d+)\s*\)",
                    txt, re.IGNORECASE | re.DOTALL
                )
                nome = m.group(1).strip() if m else fn
                cod  = m.group(2).strip() if m else fn
                # extrai conteúdo programático
                m2 = re.search(
                    r"Conte[úu]do program[aá]tico\s*[:\-–]?\s*(.*?)(?=\n\s*Bibliografia|\Z)",
                    txt, re.IGNORECASE | re.DOTALL
                )
                conteudo = m2.group(1).strip() if m2 else ""
                regs.append({
                    "COD_EMENTA": cod,
                    "NOME UC": nome,
                    "CONTEUDO_PROGRAMATICO": conteudo
                })
    return pd.DataFrame(regs)

@st.cache_data(show_spinner=False)
def get_embeddings(texts: list[str]) -> np.ndarray:
    """Gera embeddings SBERT em batches (usa o model global)."""
    return model.encode(texts, batch_size=32, convert_to_tensor=False)

@st.cache_data(show_spinner=False)
def name_cluster_with_gpt(prompt: str, api_key: str) -> str:
    """Gera nome de cluster via GPT-3.5 (cache por prompt)."""
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você resume grupos de ementas em um nome curto."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.0,
        max_tokens=20
    )
    return resp.choices[0].message.content.strip().strip('"')

# --------------------------------------------------
# 3) Upload dos arquivos
# --------------------------------------------------
uploaded_zip = st.file_uploader("📥 Faça upload do ZIP de ementas (PDF)", type="zip")
if not uploaded_zip:
    st.info("Aguardando upload do ZIP...")
    st.stop()

df_ementas = parse_ementas(uploaded_zip.read())
st.success(f"{len(df_ementas)} ementas carregadas.")
st.dataframe(df_ementas.head())

# opção de correção de pontuação
use_corr = st.sidebar.checkbox("Corrigir pontuação com ChatGPT antes de separar frases?")
api_key_corr = ""
if use_corr:
    api_key_corr = st.sidebar.text_input("Chave OpenAI para correção:", type="password")
    if api_key_corr:
        openai.api_key = api_key_corr
        with st.spinner("Corrigindo pontuação dos conteúdos..."):
            def corrige(txt: str) -> str:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role":"system","content":"Você é um corretor de pontuação de texto."},
                        {"role":"user","content":f"Corrija a pontuação deste texto, sem alterar o conteúdo:\n\n{txt}"}
                    ],
                    temperature=0.0,
                    max_tokens=len(txt.split())+50
                )
                return resp.choices[0].message.content.strip()
            df_ementas["CONTEUDO_PROGRAMATICO"] = df_ementas["CONTEUDO_PROGRAMATICO"] \
                .apply(lambda t: corrige(t) if isinstance(t,str) and t.strip() else t)
    else:
        st.sidebar.warning("Informe a API Key para habilitar correção.")

uploaded_enade = st.file_uploader("📥 Faça upload do Excel de competências ENADE", type="xlsx", key="enade")
if not uploaded_enade:
    st.info("Aguardando upload do Excel ENADE...")
    st.stop()

enade = pd.read_excel(uploaded_enade).dropna(subset=['DESCRIÇÃO'])
enade['FRASE_ENADE'] = enade['DESCRIÇÃO'].str.replace('\n',' ').str.split(r'[.;]')
enade_expl = (
    enade.explode('FRASE_ENADE')
         .assign(FRASE_ENADE=lambda df: df['FRASE_ENADE'].str.strip())
)
enade_expl = enade_expl[enade_expl['FRASE_ENADE'].str.len() > 5].reset_index(drop=True)

# --------------------------------------------------
# 4) Seleção da análise
# --------------------------------------------------
analise = st.sidebar.selectbox("Escolha a Análise", [
    "Clusterização Ementas",
    "Matriz de Similaridade",
    "Matriz de Redundância",
    "Análise Ementa vs ENADE"
])

# --------------------------------------------------
# 5) Carrega modelo SBERT em cache
# --------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = load_sbert()

# ... o resto do código permanece inalterado ...
