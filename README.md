
🕵️‍♂️ Detector de Fake News com Machine Learning
Este projeto utiliza técnicas avançadas de Processamento de Linguagem Natural (NLP) e o algoritmo Passive Aggressive Classifier para identificar notícias falsas com alta precisão. O modelo foi treinado utilizando o dataset WELFake, alcançando uma performance excepcional.

📊 Performance do Modelo
O modelo atual apresenta resultados sólidos, ideais para aplicações de verificação em tempo real:

Acurácia Geral: 95.17%

Precisão (Fake & Real): 0.95+

F1-Score: 0.95

🛠️ Tecnologias e Ferramentas
O projeto foi desenvolvido em Python, utilizando as seguintes bibliotecas:

Pandas & Scikit-learn: Para manipulação de dados e machine learning.

SpaCy: Utilizado para lematização e limpeza linguística inteligente.

TfidfVectorizer: Para transformar textos em vetores numéricos (com suporte a bi-grams).

Joblib: Para persistência (salvamento) do modelo treinado.

Tqdm: Para barras de progresso durante o processamento de grandes volumes de dados.

🚀 Como Funciona o Pipeline
1. Pré-processamento (Cleaning)
Diferente de limpezas simples, utilizamos o spaCy para:

Remoção de URLs e caracteres especiais via Regex.

Remoção de Stopwords (palavras irrelevantes como "the", "a", "is").

Lematização: Redução das palavras à sua raiz (ex: "running" vira "run"), o que ajuda o modelo a entender o significado central independentemente da conjugação.

2. Vetorização
Utilizamos o TF-IDF (Term Frequency-Inverse Document Frequency) com ngram_range=(1, 2). Isso permite que o modelo entenda não apenas palavras isoladas, mas também expressões compostas.

3. Classificação
O Passive Aggressive Classifier foi escolhido por sua eficiência em grandes volumes de dados e rapidez de resposta, sendo ideal para tarefas de classificação de texto onde a fronteira de decisão pode mudar ligeiramente.

📁 Estrutura do Repositório
fknews2.ipynb: Notebook principal com a análise, limpeza e treinamento.

modelo_fakenews_95.pkl: O modelo treinado e pronto para uso.

vetorizador_tfidf.pkl: O vetorizador necessário para transformar novos inputs.

🎮 Como Testar
No final do notebook, há um script interativo que sorteia notícias do dataset para teste.

Python

# Exemplo de saída do sistema:
📌 TÍTULO: Health Secretary Price believes has president's confidence
----------------------------------------------------------------------
📊 GABARITO (Dataset): REAL ✅
🤖 IA PREDIZ: REAL ✅
⚖️ VEREDITO: 🎯 ACERTO
