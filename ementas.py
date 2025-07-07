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
import umap                            # pip install umap-learn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import openai                          # v1.x client
import xlsxwriter                      # para formatação condicional no Excel
from xlsxwriter.utility import xl_rowcol_to_cell

# --------------------------------------------------
# 1) Configuração da página Streamlit
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("📂📑 Análise de Ementas via pasta .zip")

# --------------------------------------------------
# 2) Pergunta de correção antes do upload de ementas
# --------------------------------------------------
use_corr = st.sidebar.checkbox(
    "Corrigir pontuação das ementas com ChatGPT antes do upload?",
    help="Marque para usar GPT na correção e insira sua API Key abaixo."
)
api_key_corr = ""
if use_corr:
    api_key_corr = st.sidebar.text_input(
        "OpenAI API Key para correção:", type="password"
    )

# --------------------------------------------------
# 3) Upload do ZIP de ementas
# --------------------------------------------------
uploaded_zip = st.file_uploader(
    "📥 Faça upload do ZIP de ementas (PDF)", type="zip"
)
if not uploaded_zip:
    st.info("Aguardando upload do ZIP...")
    st.stop()

# --------------------------------------------------
# 4) Funções de parsing e correção
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
                m = re.search(
                    r"UNIDADE CURRICULAR[:\s]*(.+?)\s*\(\s*(\d+)\s*\)",
                    txt, re.IGNORECASE | re.DOTALL
                )
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
def corrige_texto(textos: list[str], api_key: str) -> list[str]:
    client = openai.OpenAI(api_key=api_key)
    out = []
    for txt in textos:
        if not isinstance(txt, str) or not txt.strip():
            out.append(txt)
            continue
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"Você corrige a pontuação deste texto sem alterar o conteúdo."},
                {"role":"user",  "content": txt}
            ],
            temperature=0.0,
            max_tokens=len(txt.split()) + 50
        )
        out.append(resp.choices[0].message.content.strip())
    return out

# --------------------------------------------------
# 5) Parse e correção
# --------------------------------------------------
df_ementas = parse_ementas(uploaded_zip.read())

if use_corr and api_key_corr:
    with st.spinner("Corrigindo pontuação dos conteúdos…"):
        textos = df_ementas["CONTEUDO_PROGRAMATICO"].tolist()
        corr = corrige_texto(textos, api_key_corr)
        df_ementas["CONTEUDO_PROGRAMATICO"] = corr
    st.subheader("📖 Conteúdos Programáticos Corrigidos")
    st.dataframe(
        df_ementas[["COD_EMENTA","NOME UC","CONTEUDO_PROGRAMATICO"]]
        .rename(columns={"CONTEUDO_PROGRAMATICO":"Conteúdo Corrigido"})
    )

st.success(f"{len(df_ementas)} ementas carregadas.")

# --------------------------------------------------
# 6) Upload do Excel ENADE
# --------------------------------------------------
uploaded_enade = st.file_uploader(
    "📥 Faça upload do Excel de competências ENADE", type="xlsx", key="enade"
)
if not uploaded_enade:
    st.info("Aguardando upload do Excel ENADE...")
    st.stop()

enade = pd.read_excel(uploaded_enade).dropna(subset=['DESCRIÇÃO'])
# explode ENADE em frases (mantém ponto+vírgula)
enade['FRASE_ENADE'] = (
    enade['DESCRIÇÃO'].str.replace('\n',' ').str.split(r'[.;]')
)
enade_expl = (
    enade.explode('FRASE_ENADE')
         .assign(FRASE_ENADE=lambda df: df['FRASE_ENADE'].str.strip())
)
enade_expl = enade_expl[enade_expl['FRASE_ENADE'].str.len() > 5].reset_index(drop=True)

# --------------------------------------------------
# 7) Escolha da análise
# --------------------------------------------------
analise = st.sidebar.selectbox("Escolha a Análise", [
    "Clusterização Ementas",
    "Matriz de Similaridade",
    "Matriz de Redundância",
    "Análise Ementa vs ENADE"
])

# --------------------------------------------------
# 8) Carrega SBERT
# --------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model = load_sbert()

# --------------------------------------------------
# 9A) Clusterização
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

    @st.cache_data
    def get_embeddings(textos):
        return model.encode(textos, batch_size=32, convert_to_tensor=False)
    emb = get_embeddings(texts)

    max_k = min(10, len(emb))
    k = st.slider("Número de clusters (K)", 2, max_k, min(4, max_k))
    km = KMeans(n_clusters=k, random_state=42).fit(emb)
    df_group['cluster'] = km.labels_

    # nomear clusters
    use_gpt = st.sidebar.checkbox("Nomear clusters com ChatGPT")
    cluster_names = {}
    if use_gpt:
        api_key_gpt = st.sidebar.text_input("OpenAI API Key para clusters:", type="password")
        if api_key_gpt:
            client_k = openai.OpenAI(api_key=api_key_gpt)
            for cid in range(k):
                exemplos = df_group[df_group['cluster']==cid]['CONTEUDO_PROGRAMATICO'].tolist()[:5]
                prompt = (
                    "Essas ementas formam um grupo temático. "
                    "Em 1 a 3 palavras, sem aspas, dê um nome que resuma o tema:\n\n"
                    + "\n".join(f"- {e}" for e in exemplos)
                )
                try:
                    resp = client_k.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role":"system","content":"Você resume conjuntos de ementas em um nome curto."},
                            {"role":"user",  "content":prompt}
                        ],
                        temperature=0.0,
                        max_tokens=10,
                    )
                    cluster_names[cid] = resp.choices[0].message.content.strip()
                except Exception as e:
                    st.error(f"Erro ao nomear cluster {cid}: {e}")
                    cluster_names[cid] = f"Cluster {cid}"
            st.write("🔖 Nomes sugeridos:", cluster_names)
        else:
            st.sidebar.warning("Informe a API Key para clusters.")
    if not use_gpt or not api_key_gpt:
        for cid in range(k):
            cent = km.cluster_centers_[cid]
            mask = np.where(km.labels_==cid)[0]
            d = np.linalg.norm(np.array(emb)[mask] - cent, axis=1)
            idx = mask[d.argmin()]
            cluster_names[cid] = df_group.at[idx, 'NOME UC']

    df_group['cluster_name'] = df_group['cluster'].map(cluster_names)

    method = st.radio("Redução de dimensão", ("PCA+t-SNE", "UMAP"))
    if method == "PCA+t-SNE":
        pca50 = PCA(n_components=min(50,len(emb)-1),random_state=42).fit_transform(emb)
        n_s = pca50.shape[0]
        perp = min(30, max(1,n_s-1))
        coords = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(pca50)
    else:
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(emb)

    df_group['X'], df_group['Y'] = coords[:,0], coords[:,1]
    fig, ax = plt.subplots(figsize=(8,6))
    pal = plt.cm.get_cmap("tab10", k)
    for cid in range(k):
        sub = df_group[df_group['cluster']==cid]
        ax.scatter(sub['X'], sub['Y'], color=pal(cid), label=cluster_names[cid], s=40, alpha=0.7)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.legend(bbox_to_anchor=(1,1))
    st.pyplot(fig)

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_group[['COD_EMENTA','NOME UC','cluster','cluster_name']].to_excel(writer, index=False, sheet_name="Clusters")
    buf.seek(0)
    st.download_button("⬇️ Baixar Clusters", buf, "clusters_ucs.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------------------------------
# 9B) Matriz de Similaridade
# --------------------------------------------------
elif analise == "Matriz de Similaridade":
    st.header("Matriz de Similaridade ENADE × Ementas")
    ementa_expl = (
        df_ementas.assign(FRASE=lambda df: df['CONTEUDO_PROGRAMATICO']
                          .str.replace('\n',' ')
                          .str.split(r'\.'))
                   .explode('FRASE')
                   .assign(FRASE=lambda df: df['FRASE'].str.strip())
    )
    ementa_expl = ementa_expl[ementa_expl['FRASE'].str.len()>5]
    with st.spinner("Calculando embeddings…"):
        emb_e = get_embeddings(ementa_expl['FRASE'].tolist())
        emb_n = get_embeddings(enade_expl['FRASE_ENADE'].tolist())
    sim = util.cos_sim(np.array(emb_n), np.array(emb_e)).cpu().numpy()
    rec, idxs = [], ementa_expl.groupby('COD_EMENTA').indices
    for cod, sidx in idxs.items():
        for i,row in enade_expl.iterrows():
            rec.append({"COD_EMENTA":cod,
                        "FRASE_ENADE":row['FRASE_ENADE'],
                        "MAX_SIM":float(sim[i,sidx].max())})
    df_sim = (pd.DataFrame(rec)
              .pivot(index='COD_EMENTA', columns='FRASE_ENADE', values='MAX_SIM')
              .fillna(0))
    st.dataframe(df_sim.style.background_gradient(cmap="RdYlGn"))
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_sim.to_excel(writer, sheet_name="Similaridade")
        wb, ws = writer.book, writer.sheets["Similaridade"]
        r,c = df_sim.shape
        start = xl_rowcol_to_cell(1,1); end = xl_rowcol_to_cell(r,c)
        ws.conditional_format(f"{start}:{end}", {
            'type':'3_color_scale','min_type':'min','min_color':"#FF0000",
            'mid_type':'percentile','mid_value':50,'mid_color':"#FFFF00",
            'max_type':'max','max_color':"#00FF00"
        })
    buf.seek(0)
    st.download_button("⬇️ Baixar Matriz", buf,
                       "sim_enade_ementa.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------------------------------
# 9C) Matriz de Redundância
# --------------------------------------------------
elif analise == "Matriz de Redundância":
    st.header("Matriz de Redundância entre Ementas")
    df_g2 = df_ementas.groupby('COD_EMENTA')['CONTEUDO_PROGRAMATICO']\
             .apply(" ".join).reset_index()
    emb2 = get_embeddings(df_g2['CONTEUDO_PROGRAMATICO'].tolist())
    sim2 = util.cos_sim(np.array(emb2), np.array(emb2)).cpu().numpy()
    df_red = pd.DataFrame(sim2, index=df_g2['COD_EMENTA'], columns=df_g2['COD_EMENTA'])
    st.dataframe(df_red.style.background_gradient(cmap="RdYlGn_r"))
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_red.to_excel(writer, sheet_name="Redundância")
        wb, ws = writer.book, writer.sheets["Redundância"]
        r,c = df_red.shape
        start = xl_rowcol_to_cell(1,1); end = xl_rowcol_to_cell(r,c)
        ws.conditional_format(f"{start}:{end}", {
            'type':'3_color_scale','min_type':'min','min_color':"#00FF00",
            'mid_type':'percentile','mid_value':50,'mid_color':"#FFFF00",
            'max_type':'max','max_color':"#FF0000"
        })
    buf.seek(0)
    st.download_button("⬇️ Baixar Redundância", buf,
                       "redundancia_uc.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --------------------------------------------------
# 9D) Análise Ementa vs ENADE
# --------------------------------------------------
else:
    st.header("Análise Ementa vs ENADE")
    df_ctx = df_ementas.copy()
    df_ctx['FRASE'] = df_ctx['CONTEUDO_PROGRAMATICO']\
                        .str.replace('\n',' ')\
                        .str.split(r'\.')
    df_ctx = df_ctx.explode('FRASE').assign(FRASE=lambda d:d['FRASE'].str.strip())
    df_ctx = df_ctx[df_ctx['FRASE'].str.len()>5].reset_index(drop=True)
    lim = st.slider("Limiar de similaridade",0.0,1.0,0.6,step=0.05)
    with st.spinner("Calculando embeddings…"):
        emb_f = get_embeddings(df_ctx['FRASE'].tolist())
        emb_n = get_embeddings(enade_expl['FRASE_ENADE'].tolist())
    simm = util.cos_sim(np.array(emb_n), np.array(emb_f)).cpu().numpy()
    recs = []
    for i,row in enade_expl.iterrows():
        sims    = simm[i]
        mx      = float(sims.max())
        imx     = int(sims.argmax())
        cmax    = df_ctx.loc[imx,'COD_EMENTA']
        tmax    = df_ctx.loc[imx,'FRASE']
        acima   = df_ctx.loc[sims>=lim,'COD_EMENTA'].unique().tolist()
        recs.append({
            "FRASE_ENADE":row['FRASE_ENADE'],
            "DIMENSÃO":row['DIMENSAO'],
            "MAX_SIM":round(mx,3),
            "COD_EMENTA_MAX":cmax,
            "TEXTO_MAX":tmax,
            f"UCs_>={int(lim*100)}%":"; ".join(map(str,acima))
        })
    df_res = pd.DataFrame(recs)
    st.subheader("Resultados por frase ENADE")
    st.dataframe(df_res)
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name="Analise_ENADE")
    buf.seek(0)
    st.download_button("⬇️ Baixar Análise vs ENADE", buf,
                       "analise_enade.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.subheader("Frequência de UCs ≥ limiar")
    col = f"UCs_>={int(lim*100)}%"
    lst = df_res[col].str.split(r';\s*').explode().dropna().astype(str)
    freq = lst.value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(freq.index,freq.values,color='skyblue')
    ax.set_xlabel("COD_EMENTA"); ax.set_ylabel("Ocorrências")
    ax.set_title(f"Ementas em ≥ {int(lim*100)}%")
    plt.xticks(rotation=45,ha='right');plt.tight_layout()
    st.pyplot(fig)
    buff=BytesIO()
    fig.savefig(buff,format='png',dpi=300,bbox_inches='tight')
    buff.seek(0)
    st.download_button("⬇️ Baixar gráfico",buff,
                       "frequencia.png","image/png")
