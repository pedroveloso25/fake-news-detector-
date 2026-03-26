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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from transformers import pipeline

# --- 1. CONFIGURAÇÕES E DESIGN ---
load_dotenv() 
st.set_page_config(page_title="Defesa Digital 2026", layout="wide", page_icon="🛡️")
api_key = os.getenv("GEMINI_API_KEY")

# Estilo experimental para melhorar a estética (Modern/Clean)
st.markdown("""
<style>
    /* Correção de visibilidade nas métricas */
    [data-testid="stMetricValue"] {
        color: #00d2ff !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        opacity: 0.9;
        font-size: 1.1rem !important;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-basetab] { font-weight: bold; }
    
    /* Melhoria no visual das abas */
    .stTabs [aria-selected="true"] {
        color: #00d2ff !important;
        border-bottom: 2px solid #00d2ff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. DOWNLOAD E CARREGAMENTO DO DATASET ---
@st.cache_resource
def setup_data():
    path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    df_path = os.path.join(path, "WELFake_Dataset.csv")
    df = pd.read_csv(df_path)
    df = df.dropna(subset=['title', 'text'])
    return df

# --- 3. ASSETS DE IA ---
@st.cache_resource
def load_ai_assets():
    # Carregamos modelos spaCy
    nlp_light = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp_full = spacy.load("en_core_web_sm")
    model = joblib.load('modelo_fakenews_95.pkl')
    tfidf = joblib.load('vetorizador_tfidf.pkl')
    # Pipeline de sentimento leve do notebook
    sent_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", truncation=True, max_length=512)
    return nlp_light, nlp_full, model, tfidf, sent_pipe

df_data = setup_data()
nlp_light, nlp_full, model, tfidf, sent_pipe = load_ai_assets()

# --- 4. PIPELINE DE LIMPEZA NLP ---
def nlp_cleaner(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    doc = nlp_light(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# --- 5. AGENTE DE EXPLICABILIDADE ---
def agente_explicador(titulo, texto, predicao):
    if not api_key:
        return "⚠️ Erro crítico: API Key não detectada."
    try:
        client = genai.Client(api_key=api_key)
        status = "Fake News" if predicao == 1 else "Notícia Real"
        prompt = f"""
        Você é um perito em desinformação digital. 
        O modelo classificou a notícia como {status}.
        Justifique essa classificação analisando padrões linguísticos e estruturais.
        TÍTULO: {titulo}
        TEXTO: {texto[:1000]}
        Seja conciso e técnico.
        """
        response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
        return response.text
    except Exception as e:
        return f"❌ Erro na comunicação com a IA: {e}"

# --- 6. FUNÇÃO DE VISUALIZAÇÃO NLP ---
def render_nlp_analysis(text):
    doc = nlp_full(text)
    
    t1, t2, t3, t4, t5 = st.tabs(["📝 Tokens", "📊 Estatísticas", "🎭 Sentimento", "🕵️ Entidades", "📏 Segmentação"])
    
    with t1:
        st.subheader("Decomposição Gramatical")
        rows = [{"Texto": t.text, "Lema": t.lemma_, "POS": t.pos_, "Dep": t.dep_} for t in list(doc)[:100]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
    with t2:
        st.subheader("Distribuição Linguística")
        c1, c2 = st.columns(2)
        with c1:
            pos_counts = pd.Series([t.pos_ for t in doc]).value_counts()
            fig, ax = plt.subplots(facecolor='none')
            sns.barplot(x=pos_counts.index, y=pos_counts.values, ax=ax, palette="viridis")
            plt.xticks(rotation=45, color='white')
            plt.yticks(color='white')
            ax.set_title("Frequência de POS Tags", color='white')
            ax.set_xlabel("Tags", color='white')
            ax.set_ylabel("Contagem", color='white')
            st.pyplot(fig)
        with c2:
            chunks = [chunk.label_ for chunk in doc.noun_chunks] if list(doc.noun_chunks) else []
            if chunks:
                chunk_counts = pd.Series(chunks).value_counts()
                fig2, ax2 = plt.subplots(facecolor='none')
                sns.barplot(x=chunk_counts.index, y=chunk_counts.values, ax=ax2, palette="magma")
                plt.xticks(color='white')
                plt.yticks(color='white')
                ax2.set_title("Tipos de Noun Chunks", color='white')
                ax2.set_xlabel("Tipos", color='white')
                ax2.set_ylabel("Contagem", color='white')
                st.pyplot(fig2)
            else: st.info("Nenhum 'Noun Chunk' complexo.")

    with t3:
        st.subheader("Análise de Sentimento (Transformers)")
        with st.container(border=True):
            with st.spinner("Analisando sentimento..."):
                res = sent_pipe(text[:512])[0]
                label_map = {"neutral": "Neutro 😐", "positive": "Positivo 🙂", "negative": "Negativo 😟"}
                sentiment_label = label_map.get(res['label'].lower(), res['label'])
                
                # Gauge ou métrica simples
                st.metric("Sentimento Identificado", sentiment_label)
                st.progress(res['score'], text=f"Confiança: {res['score']:.2%}")
            
    with t4:
        st.subheader("Entidades Nomeadas")
        if doc.ents:
            ent_html = displacy.render(doc, style="ent", jupyter=False)
            st.markdown(f'<div style="background-color: rgba(255,255,255,0.05); padding: 20px; border-radius: 10px;">{ent_html}</div>', unsafe_allow_html=True)
        else: st.info("Sem entidades detectadas.")

    with t5:
        st.subheader("Segmentação de Sentenças")
        for i, sent in enumerate(list(doc.sents)[:10]):
            st.text_area(f"Sentença {i+1}", sent.text, height=70)

# --- 7. INTERFACE PRINCIPAL ---
st.title("🛡️ Defesa Digital 2026")
st.markdown("Auditoria de Fake News e análise linguística profunda.")

# Sidebar com estatísticas
st.sidebar.title("Painel de Controle")
with st.sidebar:
    st.metric("Acurácia Experimental", "95.07%")
st.sidebar.write("---")
st.sidebar.write("**Dataset:** WELFake")
st.sidebar.write("**Análise:** spaCy v3.7+")

# Fluxo de Análise com Seleção de Fonte
source_option = st.radio("Escolha a fonte da notícia:", ["📝 Entrada Manual", "🎲 Sorteio do Dataset"], horizontal=True)

target_title = ""
target_content = ""

if source_option == "📝 Entrada Manual":
    user_text = st.text_area("Insira a notícia completa aqui para análise:", height=250)
    if st.button("🔍 Analisar Texto"):
        if user_text:
            target_content = user_text
            target_title = "Entrada Manual"
        else:
            st.warning("Insira um texto primeiro.")
else:
    if st.button("🎲 Gerar e Analisar Notícia Aleatória"):
        amostra = df_data.sample(1).iloc[0]
        target_title = amostra['title']
        target_content = amostra['text']
        st.info(f"**Título sorteado:** {target_title}")
        with st.expander("📄 Ver Conteúdo Completo da Notícia", expanded=True):
            st.write(target_content)

if target_content:
    st.markdown("---")
    # 1. Predição
    with st.spinner("Classificando notícia..."):
        limpo = nlp_cleaner(target_content)
        vetor = tfidf.transform([limpo])
        pred = model.predict(vetor)[0]
    
    # 2. Resultado Visual
    status_label = "FAKE 🚩" if pred == 1 else "REAL ✅"
    color = "#ff4b4b" if pred == 1 else "#28a745"
    st.markdown(f"""
        <div style='text-align: center; border: 2px solid {color}; padding: 15px; border-radius: 15px;'>
            <h2 style='color: {color}; margin: 0;'>Veredito: {status_label}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # 3. Análise Detalhada (Aplicações do Notebook)
    st.write("")
    with st.expander("🤖 Justificativa do Agente Consultor (XAI)", expanded=True):
        explanation = agente_explicador(target_title, target_content, pred)
        st.markdown(explanation)
        
    st.subheader("🧩 Decomposição NLP")
    render_nlp_analysis(target_content)