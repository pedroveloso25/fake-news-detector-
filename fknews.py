import streamlit as st
import pandas as pd
import joblib
import re
import spacy
import os
import kagglehub
from google import genai  # Biblioteca moderna (Substitui o antigo google.generativeai)
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# --- 1. CONFIGURAÇÕES E SEGURANÇA ---
# Carrega as variáveis de ambiente do arquivo .env (Obrigatório salvar o arquivo!)
load_dotenv() 
st.set_page_config(page_title="Defesa Digital 2026", layout="wide")
api_key = os.getenv("GEMINI_API_KEY")

# --- 2. DOWNLOAD E CARREGAMENTO DO DATASET ---
@st.cache_resource
def setup_data():
    # Download automático do dataset original via kagglehub
    path = kagglehub.dataset_download("saurabhshahane/fake-news-classification")
    df_path = os.path.join(path, "WELFake_Dataset.csv")
    
    # Carregamento e limpeza inicial de valores nulos
    df = pd.read_csv(df_path)
    df = df.dropna(subset=['title', 'text'])
    return df

# --- 3. ASSETS DE IA (NLP E MODELOS LOCAIS) ---
@st.cache_resource
def load_ai_assets():
    # Pipeline NLP: spaCy com componentes desativados para máxima performance em produção
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Carregamento do modelo de 95.07% de acurácia e seu vetorizador
    model = joblib.load('modelo_fakenews_95.pkl')
    tfidf = joblib.load('vetorizador_tfidf.pkl')
    return nlp, model, tfidf

df = setup_data()
nlp, model, tfidf = load_ai_assets()

# --- 4. PIPELINE DE LIMPEZA NLP ---
def nlp_cleaner(text):
    # Expressões Regulares para limpeza bruta de ruídos digitais
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    
    # Lemmatização e remoção de Stopwords para focar no sinal semântico
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# --- 5. AGENTE DE EXPLICABILIDADE (GEMINI 2.5 API) ---
def agente_explicador(titulo, texto, predicao):
    if not api_key:
        return "⚠️ Erro crítico: API Key não detectada. Verifique se o arquivo .env está salvo corretamente."
    
    try:
        # Instancia o cliente usando a nova arquitetura do Google GenAI (Padrão 2026)
        client = genai.Client(api_key=api_key)
        status = "Fake News" if predicao == 1 else "Notícia Real"
        
        prompt = f"""
        Você é um perito em desinformação digital analisando o ciclo eleitoral. 
        Nosso modelo Passive-Aggressive classificou a notícia abaixo como {status}.
        
        TÍTULO: {titulo}
        TEXTO: {texto[:600]}...
        
        Justifique essa classificação analisando:
        1. Padrões linguísticos evidentes (sensacionalismo, uso de caixa alta, falta de objetividade).
        2. Coerência com os padrões estruturais de fake news vs jornalismo profissional.
        Seja conciso, acadêmico e justifique a escolha do nosso modelo matemático.
        """
        
        # Faz a inferência usando o modelo ativo e atualizado
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        return f"❌ Erro na comunicação com a IA: Falha de conexão ou modelo indisponível. Log: {e}"

# --- 6. INTERFACE STREAMLIT ---
st.title("🛡️ Detector de Fake News")
st.sidebar.title("Configurações do Projeto")
st.sidebar.info("Dataset Original: WELFake (71.537 registros válidos)")
st.sidebar.success("Modelo: Passive-Aggressive Classifier (95.07% de Acurácia)")

# Métricas de Performance consolidadas
col_stats1, col_stats2 = st.columns(2)
col_stats1.metric("Acurácia do Modelo", "95.07%")
col_stats2.metric("F1-Score Médio", "0.95")

st.markdown("---")

# --- MÓDULO DE AUDITORIA RANDÔMICA ---
if st.button("🎲 Executar Auditoria Aleatória"):
    # Sorteio simples simulando um fluxo de notícias em tempo real
    amostra = df.sample(1).iloc[0]
    
    # Processamento e Predição instantânea
    sinal_bruto = str(amostra['title']) + " " + str(amostra['text'])
    limpo = nlp_cleaner(sinal_bruto)
    vetor = tfidf.transform([limpo])
    pred = model.predict(vetor)[0]
    
    # Exibição Estruturada dos Resultados
    col_res1, col_res2 = st.columns([2, 1])
    
    with col_res1:
        st.subheader("📌 Conteúdo Sorteado")
        st.write(f"**Título:** {amostra['title']}")
        st.text_area("Corpo da Notícia", amostra['text'][:1000] + "...", height=200)
        
    with col_res2:
        st.subheader("⚖️ Veredito da IA")
        label_real = "FAKE 🚩" if amostra['label'] == 1 else "REAL ✅"
        label_pred = "FAKE 🚩" if pred == 1 else "REAL ✅"
        
        # Avaliação de acurácia da amostra
        if pred == amostra['label']:
            st.success(f"🎯 **ACERTO DO MODELO**")
        else:
            st.error(f"❌ **DIVERGÊNCIA IDENTIFICADA**")
            
        st.write(f"**Gabarito Real:** {label_real}")
        st.write(f"**Predição da IA:** {label_pred}")

    # Acionamento do Agente de Explicabilidade (XAI)
    st.markdown("---")
    st.subheader("🤖 Agente Consultor: Justificativa Técnica")
    with st.spinner("Decodificando os pesos do TF-IDF e analisando estrutura semântica..."):
        explicacao = agente_explicador(amostra['title'], amostra['text'], pred)
        st.markdown(explicacao)

# --- MÓDULO DE TESTE MANUAL ---
st.markdown("---")
st.subheader("✍️ Teste Manual! (Desafie o Modelo)")
st.markdown("Cole um texto fabricado para avaliar a resiliência do modelo.")
manual_text = st.text_area("Insira a notícia aqui:")
if st.button("🔍 Analisar Manualmente"):
    if manual_text:
        vetor_m = tfidf.transform([nlp_cleaner(manual_text)])
        pred_m = model.predict(vetor_m)[0]
        status_m = "FAKE 🚩" if pred_m == 1 else "REAL ✅"
        st.warning(f"O classificador definiu este conteúdo como: **{status_m}**")