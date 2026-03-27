# 🕵️‍♂️ Detector de Fake News com Machine Learning + NLP Avançado

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/Licença-MIT-green?style=for-the-badge)

**Classificador de notícias falsas com 95%+ de acurácia, combinando NLP clássico, Transformers e análise linguística profunda sobre o dataset WELFake.**

</div>

---

## 📌 Sobre o Projeto

Este projeto aplica um pipeline completo de **Processamento de Linguagem Natural (NLP)** para identificar automaticamente se uma notícia é **real ou falsa**. O trabalho é dividido em etapas progressivas: da limpeza e modelagem clássica com ML até análise semântica profunda com **Transformers** e **Sentence Embeddings**.

O corpus utilizado é o **WELFake Dataset** — mais de 70.000 artigos rotulados — disponível no Kaggle.

---

## 📊 Desempenho dos Modelos

| Modelo | Acurácia |
|---|---|
| Passive Aggressive Classifier | **~95%** |
| Linear SVC | **~95%** |
| Logistic Regression | **~94%** |
| Naive Bayes | **~93%** |

> Todos os modelos treinados com TF-IDF (bigramas, 5.000 features) sobre o dataset completo.

---

## 🗂️ Estrutura do Repositório

```
fake-news-detector/
│
├── fknews2.ipynb             # Etapa 2: Treinamento e avaliação do modelo principal
├── fknews3.ipynb             # Etapa 3: NLP avançado, Transformers e Embeddings
├── fknews.py                 # Script Python standalone
├── modelo_fakenews_95.pkl    # Modelo treinado (pronto para uso)
├── vetorizador_tfidf.pkl     # Vetorizador TF-IDF serializado
├── requirements.txt          # Dependências do projeto
└── README.md
```

---

## 🔬 Pipeline Completo

### Etapa 2 — Modelagem Clássica (`fknews2.ipynb`)

```
Texto bruto → Limpeza (spaCy + Regex) → TF-IDF Bigramas → Classificador → REAL ✅ / FAKE ❌
```

**Pré-processamento com spaCy:**
- Remoção de URLs e caracteres especiais via Regex
- Remoção de stopwords
- Lematização (ex: "running" → "run")

**Vetorização:**
- TF-IDF com `ngram_range=(1,2)` e `sublinear_tf=True`
- Captura palavras isoladas e expressões compostas

**Classificador principal:** Passive Aggressive Classifier — eficiente, robusto a ruído, ideal para texto esparso de alta dimensionalidade.

---

### Etapa 3 — NLP Avançado e Análise Linguística (`fknews3.ipynb`)

Análise sobre uma amostra balanceada de **200 notícias (100 FAKE + 100 REAL)** com o pipeline completo do spaCy.

#### 1. Tokenização, Normalização e Lematização
Comparação entre tokens brutos e lematizados por classe. **Achado:** notícias FAKE apresentam mais tokens brutos, mas após lematização o tamanho semântico se iguala — o excesso de extensão vinha de stopwords, não de conteúdo.

#### 2. POS Tagging (Part-of-Speech)
Identificação das categorias gramaticais (substantivos, verbos, adjetivos...). **Achado:** FAKE news usam mais **adjetivos e advérbios** (linguagem emocional); REAL news usam mais **nomes próprios** (linguagem factual).

#### 3. Noun Chunks (Sintagmas Nominais)
Extração de expressões compostas com significado unitário (ex: "the fake news media"). **Achado:** chunks de FAKE news orbitam em torno de figuras políticas com tom sensacionalista; REAL news apontam para instituições e processos.

#### 4. NER — Reconhecimento de Entidades Nomeadas
Classificação automática de pessoas, organizações, países, datas e eventos. **Achado:** REAL news citam mais organizações (`ORG`) e veículos de imprensa (Reuters, NYT); FAKE news citam mais pessoas (`PERSON`) sem fontes jornalísticas consolidadas.

#### 5. Segmentação de Sentenças
Análise do número e comprimento médio de sentenças por classe. **Achado:** FAKE news têm mais sentenças por notícia, porém mais curtas — estilo fragmentado e exclamativo que imita manchetes de impacto. REAL news têm menos sentenças, porém mais longas e informativas.

#### 6. Comparação de Classificadores
Treinamento e avaliação de 4 modelos com TF-IDF sobre o dataset completo. **Achado:** PAC e SVM alcançam ~95%; modelos lineares que lidam bem com esparsidade são superiores para texto.

#### 7. Análise de Sentimentos com Transformers
Uso do modelo `cardiffnlp/twitter-roberta-base-sentiment-latest` (HuggingFace, treinado em ~58M tweets) para classificar o sentimento de cada notícia. **Achado:** FAKE news apresentam maior proporção de sentimento **negativo**; REAL news são predominantemente **neutras**. O sentimento atua como feature auxiliar, mas é insuficiente isoladamente.

#### 8. Embeddings Semânticos com Sentence Transformers
Geração de embeddings com `all-MiniLM-L6-v2` (384 dimensões) e visualização da separação FAKE vs REAL via **PCA** e **t-SNE**. **Achado:** há agrupamento visual parcial entre as classes no espaço vetorial, confirmando que o significado semântico dos títulos carrega sinal discriminativo.

---

## 📋 Tabela de Achados (Etapa 3)

| Análise | Principal Achado |
|---|---|
| Tokenização / Lematização | FAKE news têm mais tokens brutos; após lematização o tamanho semântico se iguala |
| POS Tagging | FAKE: mais adjetivos e advérbios; REAL: mais nomes próprios |
| Noun Chunks | FAKE: foco em figuras políticas; REAL: foco em instituições e processos |
| NER | REAL: mais `ORG` e imprensa consolidada; FAKE: mais `PERSON` |
| Segmentação | FAKE: sentenças curtas e numerosas; REAL: sentenças longas e informativas |
| Classificação | Todos os modelos superam 93%; PAC e SVC alcançam ~95% |
| Sentimentos | FAKE: mais negativo; REAL: predominantemente neutro |
| Embeddings | Agrupamento parcial visível em PCA e t-SNE |

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Finalidade |
|---|---|
| `pandas` / `numpy` | Manipulação e análise de dados |
| `scikit-learn` | Modelos ML, TF-IDF, PCA, t-SNE |
| `spaCy` | Tokenização, POS, NER, Noun Chunks, Segmentação |
| `transformers` (HuggingFace) | Análise de sentimentos com RoBERTa |
| `sentence-transformers` | Embeddings semânticos com MiniLM |
| `matplotlib` / `seaborn` | Visualizações |
| `wordcloud` | Nuvens de palavras dos Noun Chunks |
| `joblib` | Serialização do modelo treinado |
| `tqdm` | Barras de progresso |

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/pedroveloso25/fake-news-detector-.git
cd fake-news-detector-
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixe o modelo de linguagem do spaCy

```bash
python -m spacy download en_core_web_sm
```

### 4. Execute os notebooks

```bash
# Etapa 2 — Treinamento do modelo
jupyter notebook fknews2.ipynb

# Etapa 3 — NLP avançado e Transformers
jupyter notebook fknews3.ipynb
```

> **Nota:** A Etapa 3 requer GPU ou paciência — o modelo Transformer e os embeddings podem levar alguns minutos na CPU.

---

## 🎮 Exemplo de Saída (Etapa 2)

```
📌 TÍTULO: Health Secretary Price believes has president's confidence

📊 GABARITO (Dataset): REAL ✅
🤖 IA PREDIZ:          REAL ✅
⚖️  VEREDITO:          🎯 ACERTO
```

---

## 📂 Dataset

**[WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)** — disponível no Kaggle. Benchmark amplamente utilizado em pesquisas de detecção de desinformação, com mais de 70.000 artigos rotulados como reais ou falsos.

---

## 👨‍💻 Autores

**Pedro Veloso**  
Estudante de Ciência de Dados para Negócios — UFPB  
Trainee de Data & AI — TAIL (Technology and Artificial Intelligence League)

**Vitor Batista**
Estudante de Ciência de Dados para Negócios — UFPB  
Membro da LMFUFPB (Liga de mercado financeiro UFPB)

**Marcello Siqueira**
Estudante de Ciência de Dados para Negócios — UFPB  
Analista de dados na energisa 

[![GitHub](https://img.shields.io/badge/GitHub-pedroveloso25-181717?style=flat-square&logo=github)](https://github.com/pedroveloso25)

---

## 📄 Licença

Este projeto está sob a licença **MIT**. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
