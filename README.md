# 🕵️‍♂️ Detector de Fake News com Machine Learning + Agente IA

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://detectorfakenews.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Acurácia](https://img.shields.io/badge/Acur%C3%A1cia-95.17%25-brightgreen)](#-performance-do-modelo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **🚀 Demo ao vivo:** [detectorfakenews.streamlit.app](https://detectorfakenews.streamlit.app)

Projeto de detecção de fake news que combina um modelo de Machine Learning treinado com o dataset WELFake com um **Agente Consultor de IA (Google Gemini)** que fornece uma justificativa técnica detalhada para cada classificação. Conta também com um modo de **Teste Manual** onde você pode colar qualquer notícia e receber um veredito instantâneo.

---

## ✨ Novidades (v2.0)

- 🤖 **Agente Consultor (Gemini API):** após a classificação pelo modelo, um agente de IA analisa heurísticas linguísticas da notícia — sensacionalismo, linguagem pejorativa, ausência de fontes, descontextualização — e gera uma justificativa técnica detalhada.
- ✍️ **Teste Manual:** cole o corpo de qualquer notícia diretamente na interface e desafie o modelo em tempo real.
- 🌐 **Interface Web com Streamlit:** totalmente acessível pelo navegador, sem necessidade de instalação local.

---

## 📊 Performance do Modelo

| Métrica | Valor |
|---|---|
| Acurácia Geral | **95.17%** |
| Precisão (Fake & Real) | 0.95+ |
| F1-Score | 0.95 |

---

## 🛠️ Tecnologias

| Camada | Ferramenta |
|---|---|
| Linguagem | Python 3.9+ |
| ML / NLP | Scikit-learn, SpaCy, TF-IDF |
| Classificador | Passive Aggressive Classifier |
| Agente IA | Google Gemini API |
| Interface | Streamlit |
| Persistência | Joblib (.pkl) |
| Dataset | WELFake |

---

## 🚀 Como Funciona

### 1. Pré-processamento
Utilizamos o **spaCy** para uma limpeza linguística inteligente:
- Remoção de URLs e caracteres especiais via Regex
- Remoção de Stopwords
- **Lematização:** redução das palavras à sua raiz (ex: `"running"` → `"run"`)

### 2. Vetorização
**TF-IDF** com `ngram_range=(1, 2)` — o modelo entende não apenas palavras isoladas, mas expressões compostas, captando padrões linguísticos mais ricos.

### 3. Classificação
O **Passive Aggressive Classifier** foi escolhido pela sua eficiência em grandes volumes de texto e rapidez de resposta em tarefas de classificação binária.

### 4. Agente Consultor (Gemini)
Após a predição, a **Gemini API** analisa a notícia e gera uma justificativa técnica baseada em marcadores heurísticos de desinformação:
- Sensacionalismo e imperatividade no título
- Linguagem pejorativa e partidária
- Ausência de fontes verificáveis
- Generalizações não comprováveis
- Citações incompletas ou descontextualizadas

---

## 📁 Estrutura do Repositório

```
fake-news-detector/
├── fknews.py                  # App principal (Streamlit)
├── fknews2.ipynb              # Notebook de análise e treinamento
├── modelo_fakenews_95.pkl     # Modelo treinado (Passive Aggressive)
├── vetorizador_tfidf.pkl      # Vetorizador TF-IDF serializado
├── requirements.txt           # Dependências do projeto
└── README.md
```

---

## ⚙️ Como Rodar Localmente

```bash
# 1. Clone o repositório
git clone https://github.com/pedroveloso25/fake-news-detector-
cd fake-news-detector-

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Configure sua chave da Gemini API
# Crie um arquivo .env ou defina a variável de ambiente:
export GEMINI_API_KEY="sua_chave_aqui"

# 4. Rode a aplicação
streamlit run fknews.py
```

---

## 🎮 Funcionalidades da Interface

### Classificação Automática
O sistema sorteia notícias do dataset para demonstrar o modelo em ação:
```
📌 TÍTULO: Health Secretary Price believes has president's confidence
📊 GABARITO (Dataset): REAL ✅
🤖 IA PREDIZ:          REAL ✅
⚖️ VEREDITO:           🎯 ACERTO
```

### Teste Manual
Cole o texto de qualquer notícia e receba:
1. **Classificação do modelo** — FAKE ou REAL
2. **Justificativa técnica do Agente Gemini** com análise dos padrões linguísticos detectados

---

## 🔮 Próximos Passos

- [ ] Suporte a notícias em português (PT-BR)
- [ ] Histórico de análises por sessão
- [ ] Score de confiança por heurística
- [ ] Integração com APIs de fact-checking

---

## 👤 Autor

**Pedro Veloso** — Estudante de Ciência de Dados para Negócios @ UFPB

[![GitHub](https://img.shields.io/badge/GitHub-pedroveloso25-181717?logo=github)](https://github.com/pedroveloso25)
