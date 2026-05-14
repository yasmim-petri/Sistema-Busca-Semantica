
import json
import pandas as pd
import re
import html
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Caminhos e modelo utilizados
noticia_json = r"dados/noticias_brutas.json"
noticias_limpas = "dados/noticias_limpas.csv"
model_name = 'ibm-granite/granite-embedding-97m-multilingual-r2'
caminho_embeddings = "dados/embeddings_noticias.pkl"

def limpar_texto_noticia(texto):
    if not isinstance(texto, str): return ""
    texto = html.unescape(texto)
    texto = re.sub(r'<[^>]+>', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# Limpeza e processamento dos dados
def processar_dados():
    if not os.path.exists(noticia_json):
        print(f"Erro: {noticia_json} não encontrado.")
        return False
    
    os.makedirs("dados", exist_ok=True)
    with open(noticia_json, 'r', encoding='utf-8') as f: # Carrega o JSON diretamente para um DataFrame
        dados = json.load(f)
    
    df = pd.DataFrame(dados)
    df.columns = df.columns.str.strip()
    
    if 'texto' in df.columns:
        df['texto'] = df['texto'].apply(limpar_texto_noticia)
    
    estrutura = ["id", "titulo", "texto", "data", "fonte"]
    colunas_validas = [c for c in estrutura if c in df.columns] # Garante que apenas as colunas válidas sejam salvas
    df[colunas_validas].to_csv(noticias_limpas, index=False, encoding='utf-8')
    print("Dados limpos e salvos.")
    return True

if processar_dados():
    # Geração de embeddings
    model = SentenceTransformer(model_name)
    df = pd.read_csv(noticias_limpas)
    
    # Para uma busca mais rica utilizei título e texto
    textos_para_processar = (df['titulo'].fillna('') + " " + df['texto'].fillna('')).tolist()
    
    print(f"Gerando embeddings para {len(textos_para_processar)} noticias")
    embeddings = model.encode(textos_para_processar, batch_size=32, show_progress_bar=True) 

    # Salva o embedding como um dicionário para facilitar a leitura depois
    data_to_save = {
        'embeddings': embeddings,
        'textos': df['texto'].tolist(),
        'titulos': df['titulo'].tolist()
    }
    
    with open(caminho_embeddings, 'wb') as f: # Salva o dicionário completo
        pickle.dump(data_to_save, f)
    print(f"Arquivo {caminho_embeddings} salvo com sucesso.")

    # Carregar os dados corretamente
    with open(caminho_embeddings, "rb") as f:
        data_carregada = pickle.load(f)
        doc_embeddings = data_carregada['embeddings']
        documentos = data_carregada['textos']
        titulos = data_carregada['titulos']

    query = "mudanças na taxa de juros"
    query_embedding = model.encode([query])

    # Cálculo de similaridade
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Pegar os top 5 índices
    results_idx = np.argsort(similarities)[::-1][:5]

    print(f"\n Consultar: {query}")
    for idx in results_idx:
        score = similarities[idx]
        print(f" Score: {score:.4f}") # Exibir o score de similaridade para cada resultado
        print(f" ID: {idx}")
        print(f" Titulo: {titulos[idx]}")
        print(f" Texto: {documentos[idx][:300]}...") # Limitar a exibição do texto para os primeiros 300 caracteres
        print(f" Fonte: {df['fonte'][idx]}")
        print("-" * 50)