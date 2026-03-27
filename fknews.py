import streamlit as st
import pandas as pd
import joblib
import re
import spacy
from spacy import displacy
import os
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
from google import genai
from dotenv import load_dotenv
from wordcloud import WordCloud
import subprocess
import sys

# --- 1. CONFIGURAÇÕES E ESTILO ---
load_dotenv() 
st.set_page_config(
    page_title="DEFESA DIGITAL | Auditoria de Desinformação", 
    layout="wide", 
    page_icon="🛡️"
)

# Mantendo seu CSS Premium
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; }
    [data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 800; }
    [data-testid="metric-container"] { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px; }
    .result-card { padding: 30px; border-radius: 20px; text-align: center; margin: 20px 0; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
    .stSidebar { background-color: rgba(15, 23, 42, 0.8) !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. CARREGAMENTO DE ASSETS (CORRIGIDO) ---
@st.cache_resource
def load_resources():
    # Prevenção de erro E050: Instala o modelo se não existir
    try:
        nlp_sm = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp_sm = spacy.load("en_core_web_sm")

    # Carregamento do Dataset via KaggleHub
    # Nota: Se o deploy travar aqui, reduza para um CSV local de amostra.
    path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    df = pd.read_csv(os.path.join(path, "WELFake_Dataset.csv")).dropna(subset=['title', 'text'])
    
    # Carregamento dos modelos treinados
    model = joblib.load('modelo_fakenews_95.pkl')
    tfidf = joblib.load('vetorizador_tfidf.pkl')
    
    return df, nlp_sm, model, tfidf

with st.spinner("🚀 Carregando Sistemas de Defesa..."):
    try:
        df_data, nlp, model, tfidf = load_resources()
    except Exception as e:
        st.error(f"Erro crítico ao carregar recursos: {e}")
        st.stop()

# --- 3. LÓGICA DE LIMPEZA ---
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    return text

# --- 4. PAINEL DE CONTROLE (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2092/2092663.png", width=80)
    st.title("Painel de Controle")
    st.markdown("---")
    
    st.subheader("📊 Estatísticas do Modelo")
    st.metric("Acurácia (TF-IDF)", "95.07%")
    st.metric("Tempo/Análise", "0.4s")
    
    st.markdown("---")
    
    with st.expander("🔬 Insights do Dataset (Geral)", expanded=False):
        st.write("Análise de amostra do WELFake")
        
        # Amostragem otimizada para não estourar a memória
        sample_df = df_data['title'].sample(min(100, len(df_data)))
        sample_text = " ".join(sample_df)
        
        wc = WordCloud(background_color=None, mode="RGBA", colormap="Blues").generate(sample_text)
        fig_wc, ax_wc = plt.subplots()
        ax_wc.imshow(wc); ax_wc.axis("off"); fig_wc.patch.set_alpha(0)
        st.pyplot(fig_wc)
        
        st.markdown("**Principais Termos**")
        st.bar_chart(sample_df.str.split().str[0].value_counts().head(10))

# --- 5. CABEÇALHO (MAIN) ---
c1, c2 = st.columns([1, 5])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/1162/1162916.png", width=120)
with c2:
    st.title("SISTEMA DEFESA DIGITAL 2026")
    st.markdown("*Monitoramento Avançado de Integridade de Informação*")

source = st.selectbox("Selecione a fonte de dados:", ["📝 Texto Livre", "🎲 Selecionar amostra do Dataset"])

target_title, target_content = "", ""

if source == "📝 Texto Livre":
    user_input = st.text_area("Insira o artigo para auditoria:", height=150)
    if st.button("🔍 ANALISAR TEXTO") and user_input:
        target_content, target_title = user_input, "Entrada Individual"
elif source == "🎲 Selecionar amostra do Dataset":
    if st.button("🎲 SORTEAR NOTÍCIA"):
        amostra = df_data.sample(1).iloc[0]
        target_title, target_content = amostra['title'], amostra['text']
        st.success(f"**Analisando:** {target_title}")
        with st.expander("📄 Ler Notícia Completa", expanded=True):
            st.write(target_content)

# --- 6. RESULTADOS ---
if target_content:
    with st.spinner("⚙️ Processando Forense..."):
        texto_limpo = clean_text(target_content)
        # Garante que o input seja transformado corretamente pelo TF-IDF
        pred = model.predict(tfidf.transform([texto_limpo]))[0]
    
    color = "#ef4444" if pred == 1 else "#10b981"
    verdict = "🚩 ALERTA: FAKE NEWS" if pred == 1 else "✅ VERIFICADO: REAL"
    
    st.markdown(f"""
        <div class="result-card" style="border-color: {color}; background-color: rgba({('239,68,68' if pred==1 else '16,185,129')}, 0.1);">
            <h1 style="color: {color}; margin: 0;">{verdict}</h1>
        </div>
    """, unsafe_allow_html=True)
    
    doc = nlp(target_content[:3000]) # Limitado para evitar lentidão no visualizador
    tab_xai, tab_nlp = st.tabs(["🤖 Justificativa (XAI)", "🧩 Detalhes NLP (Atual)"])
    
    with tab_xai:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                client = genai.Client(api_key=api_key)
                prompt = f"Analise por que esta notícia é {'Fake' if pred==1 else 'Real'}. Título: {target_title}. Texto: {target_content[:1000]}"
                expl = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
                st.markdown(expl.text)
            except Exception as e:
                st.error("⚠️ Erro na API do Gemini.")
        else: 
            st.warning("Configure o Gemini no .env para ver a justificativa.")
        
    with tab_nlp:
        col_nlp1, col_nlp2 = st.columns(2)
        with col_nlp1: 
            st.markdown("**Entidades Identificadas**")
            if doc.ents:
                html = displacy.render(doc, style="ent", jupyter=False)
                st.markdown(html, unsafe_allow_html=True)
            else: st.info("Nenhuma entidade detectada.")
        with col_nlp2:
            st.markdown("**Gramática (POS Tags)**")
            pos_counts = pd.Series([t.pos_ for t in doc]).value_counts()
            fig_p, ax_p = plt.subplots(); fig_p.patch.set_alpha(0); ax_p.set_facecolor("none")
            sns.barplot(x=pos_counts.index, y=pos_counts.values, ax=ax_p, palette="Blues_d")
            plt.xticks(rotation=45, color="white"); plt.yticks(color="white")
            st.pyplot(fig_p)

st.markdown("---")
st.caption("© 2026 Defesa Digital | UFPB")
