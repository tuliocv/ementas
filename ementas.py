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
import umap                    # import correto do umap-learn
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
# 2) Funções com cache para etapas pesadas
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def parse_ementas(zip_bytes: bytes) -> pd.DataFrame:
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
                txt = re.sub(r"(?m)^\s*\d+\s+de\s+\d+\s*$", "", txt)
                m = re.search(r"UNIDADE CURRICULAR[:\s]*(.+?)\s*\(\s*(\d+)\s*\)",
                              txt, re.IGNORECASE | re.DOTALL)
                nome = m.group(1).strip() if m else fn
                cod  = m.group(2).strip() if m else fn
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
def get_embeddings(texts: list[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, convert_to_tensor=False)

@st.cache_data(show_spinner=False)
def name_cluster_with_gpt(prompt: str, api_key: str) -> str:
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
# 3) Upload ZIP de ementas
# --------------------------------------------------
uploaded_zip = st.file_uploader("📥 Faça upload do ZIP de ementas (PDF)", type="zip")
if not uploaded_zip:
    st.info("Aguardando upload do ZIP...")
    st.stop()

df_ementas = parse_ementas(uploaded_zip.read())
st.success(f"{len(df_ementas)} ementas carregadas.")
st.dataframe(df_ementas.head())

# --------------------------------------------------
# 4) Upload Excel ENADE
# --------------------------------------------------
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
# 5) Escolha da análise
# --------------------------------------------------
analise = st.sidebar.selectbox("Escolha a Análise", [
    "Clusterização Ementas",
    "Matriz de Similaridade",
    "Matriz de Redundância",
    "Análise Ementa vs ENADE"
])

# --------------------------------------------------
# 6) Carrega modelo SBERT em cache
# --------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = load_sbert()

# --------------------------------------------------
# 6A) Clusterização das Ementas
# --------------------------------------------------
if analise == "Clusterização Ementas":
    st.header("Clusterização das UCs")
    df_group = (
        df_ementas
        .groupby(['COD_EMENTA','NOME UC'])['CONTEUDO_PROGRAMATICO']
        .apply(" ".join)
        .reset_index()
    )
    texts = df_group['CONTEUDO_PROGRAMATICO'].tolist()
    emb    = get_embeddings(texts, model)

    max_k = min(10, len(emb))
    k = st.slider("Número de clusters (K)", 2, max_k, min(4, max_k))
    km = KMeans(n_clusters=k, random_state=42).fit(emb)
    df_group['cluster'] = km.labels_

    use_gpt = st.checkbox("Nomear clusters com GPT-3.5", False)
    api_key = ""
    if use_gpt:
        api_key = st.text_input("OpenAI API Key:", type="password")
    cluster_names = {}
    for cid in range(k):
        if use_gpt and api_key:
            exemplos = df_group[df_group['cluster']==cid]['CONTEUDO_PROGRAMATICO'].tolist()[:5]
            prompt   = "Estas são ementas de um mesmo grupo:\n" + "\n".join(f"- {e}" for e in exemplos)
            prompt  += "\nDê um nome curto (até 3 palavras) que resuma o tema."
            try:
                cluster_names[cid] = name_cluster_with_gpt(prompt, api_key)
            except:
                cluster_names[cid] = f"Cluster {cid}"
        else:
            cent = km.cluster_centers_[cid]
            mask = np.where(km.labels_==cid)[0]
            dist = np.linalg.norm(np.array(emb)[mask] - cent, axis=1)
            idx  = mask[dist.argmin()]
            cluster_names[cid] = df_group.at[idx, 'NOME UC']
    df_group['cluster_name'] = df_group['cluster'].map(cluster_names)

    method = st.radio("Redução de dimensão", ("PCA+t-SNE", "UMAP"))
    if method == "PCA+t-SNE":
        pca50 = PCA(n_components=min(50, len(emb)-1), random_state=42).fit_transform(emb)
        coords = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(pca50)
    else:
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(emb)
    df_group[['X','Y']] = coords[:,0], coords[:,1]

    fig, ax = plt.subplots(figsize=(8,6))
    palette = plt.cm.get_cmap("tab10", k)
    for cid in range(k):
        sub = df_group[df_group['cluster']==cid]
        ax.scatter(sub['X'], sub['Y'], color=palette(cid), label=cluster_names[cid], s=40, alpha=0.7)
    ax.set_xlabel("Dimensão 1"); ax.set_ylabel("Dimensão 2")
    ax.legend(title="Clusters", bbox_to_anchor=(1,1))
    st.pyplot(fig)

    df_out = df_group[['COD_EMENTA','NOME UC','cluster','cluster_name']]
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_out.to_excel(writer, index=False, sheet_name="Clusters")
        writer.save()
    buf.seek(0)
    st.download_button("⬇️ Baixar Clusters", buf, "clusters_ucs.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------------------------------
# 6B) Matriz de Similaridade
# --------------------------------------------------
elif analise == "Matriz de Similaridade":
    st.header("Matriz de Similaridade ENADE × Ementas")
    ementa_expl = (
        df_ementas
        .assign(FRASE=lambda df: df['CONTEUDO_PROGRAMATICO']
                .str.replace('\n',' ')
                .str.split(r'[.;]'))
        .explode('FRASE')
        .assign(FRASE=lambda df: df['FRASE'].str.strip())
    )
    ementa_expl = ementa_expl[ementa_expl['FRASE'].str.len()>5]

    with st.spinner("Calculando embeddings…"):
        emb_e = get_embeddings(ementa_expl['FRASE'].tolist(), model)
        emb_n = get_embeddings(enade_expl['FRASE_ENADE'].tolist(), model)

    sim = util.cos_sim(np.array(emb_n), np.array(emb_e)).cpu().numpy()
    rec = []
    idxs = ementa_expl.groupby('COD_EMENTA').indices
    for cod, sidx in idxs.items():
        for i, row in enade_expl.iterrows():
            rec.append({
                "COD_EMENTA": cod,
                "FRASE_ENADE": row['FRASE_ENADE'],
                "MAX_SIM": float(sim[i, sidx].max())
            })
    df_sim = (
        pd.DataFrame(rec)
          .pivot(index='COD_EMENTA', columns='FRASE_ENADE', values='MAX_SIM')
          .fillna(0)
    )
    st.dataframe(df_sim.style.background_gradient(cmap="RdYlGn"))

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_sim.to_excel(writer, sheet_name="Similaridade")
        wb  = writer.book
        ws  = writer.sheets["Similaridade"]
        nrows, ncols = df_sim.shape
        start = xl_rowcol_to_cell(1, 1)
        end   = xl_rowcol_to_cell(nrows, ncols)
        ws.conditional_format(f"{start}:{end}", {
            'type':     '3_color_scale',
            'min_type': 'min',    'min_color': "#FF0000",
            'mid_type': 'percentile','mid_value':50,'mid_color': "#FFFF00",
            'max_type': 'max',    'max_color': "#00FF00"
        })
        writer.save()
    buf.seek(0)
    st.download_button("⬇️ Baixar Matriz de Similaridade",
                       buf, "sim_enade_ementa_colorido.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------------------------------
# 6C) Matriz de Redundância
# --------------------------------------------------
elif analise == "Matriz de Redundância":
    st.header("Matriz de Redundância entre Ementas")
    df_group = (
        df_ementas
        .groupby('COD_EMENTA')['CONTEUDO_PROGRAMATICO']
        .apply(" ".join)
        .reset_index()
    )
    emb = get_embeddings(df_group['CONTEUDO_PROGRAMATICO'].tolist(), model)
    sim = util.cos_sim(np.array(emb), np.array(emb)).cpu().numpy()
    df_red = pd.DataFrame(sim,
                          index=df_group['COD_EMENTA'],
                          columns=df_group['COD_EMENTA'])
    st.dataframe(df_red.style.background_gradient(cmap="RdYlGn_r"))

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_red.to_excel(writer, sheet_name="Redundância")
        wb  = writer.book
        ws  = writer.sheets["Redundância"]
        nrows, ncols = df_red.shape
        start = xl_rowcol_to_cell(1, 1)
        end   = xl_rowcol_to_cell(nrows, ncols)
        ws.conditional_format(f"{start}:{end}", {
            'type':     '3_color_scale',
            'min_type': 'min',    'min_color': "#00FF00",
            'mid_type': 'percentile','mid_value':50,'mid_color': "#FFFF00",
            'max_type': 'max',    'max_color': "#FF0000"
        })
        writer.save()
    buf.seek(0)
    st.download_button("⬇️ Baixar Matriz de Redundância",
                       buf, "redundancia_uc_colorida.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------------------------------
# 6D) Análise Ementa vs ENADE
# --------------------------------------------------
else:
    st.header("Análise Ementa vs ENADE")
    df_ctx = df_ementas.copy()
    df_ctx['FRASE'] = (
        df_ctx['CONTEUDO_PROGRAMATICO']
          .str.replace('\n',' ')
          .str.split(r'\.')
    )
    df_ctx = (
        df_ctx.explode('FRASE')
              .assign(FRASE=lambda d: d['FRASE'].str.strip())
    )
    df_ctx = df_ctx[df_ctx['FRASE'].str.len() > 5].reset_index(drop=True)

    limiar = st.slider("Limiar de similaridade", 0.0, 1.0, 0.6, step=0.05)
    with st.spinner("Calculando embeddings…"):
        emb_f = get_embeddings(df_ctx['FRASE'].tolist(), model)
        emb_n = get_embeddings(enade_expl['FRASE_ENADE'].tolist(), model)
    simm = util.cos_sim(np.array(emb_n), np.array(emb_f)).cpu().numpy()

    records = []
    for i, row in enade_expl.iterrows():
        sims    = simm[i]
        max_sim = float(sims.max())
        idx_max = int(sims.argmax())
        cod_max = df_ctx.loc[idx_max, 'COD_EMENTA']
        text_max= df_ctx.loc[idx_max, 'FRASE']
        above   = df_ctx.loc[sims >= limiar, 'COD_EMENTA'].unique().tolist()
        records.append({
            "FRASE_ENADE":     row['FRASE_ENADE'],
            "DIMENSÃO":        row['DIMENSAO'],
            "MAX_SIM":         round(max_sim, 3),
            "COD_EMENTA_MAX":  cod_max,
            "TEXTO_MAX":       text_max,
            f"UCs_>={int(limiar*100)}%": "; ".join(map(str, above))
        })
    df_res = pd.DataFrame(records)
    st.subheader("Resultados por frase ENADE")
    st.dataframe(df_res)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name="Analise_ENADE")
        writer.save()
    buf.seek(0)
    st.download_button("⬇️ Baixar Análise vs ENADE",
                       buf, "analise_ementa_vs_enade.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.subheader("Frequência de UCs ≥ limiar")
    col_uc = f"UCs_>={int(limiar*100)}%"
    lista_cod = (
        df_res[col_uc]
        .str.split(r';\s*')
        .explode()
        .dropna()
        .astype(str)
    )
    freq = lista_cod.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(freq.index, freq.values, color='skyblue')
    ax.set_xlabel("COD_EMENTA")
    ax.set_ylabel("Ocorrências")
    ax.set_title(f"Ementas em ≥ {int(limiar*100)}% de similaridade")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

    buf_fig = BytesIO()
    fig.savefig(buf_fig, format="png", dpi=300, bbox_inches="tight")
    buf_fig.seek(0)
    st.download_button("⬇️ Baixar gráfico de frequência",
                       buf_fig, "frequencia_ementas.png",
                       "image/png")