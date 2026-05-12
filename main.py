import json
import pandas as pd
import re
import html
import os

# Caminhos dos arquivos utilizados
noticia_json = r"dados\noticias_brutas.json"
noticia_limpa = "dados/noticias_limpas.csv"
noticia_organizada = "dados/noticias_formatadas_final.csv"

# Função para limpar o texto das notícias

def limpar_texto_noticia(texto):
    """Realiza a limpeza de HTML, entidades e espaços extras."""
    if not isinstance(texto, str):
        return ""
    
    # Decodificar entidades HTML (&aacute;, etc)
    texto = html.unescape(texto)
    # Remover tags HTML
    texto = re.sub(r'<[^>]+>', ' ', texto)
    # Normalizar espaços e quebras de linha
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto.strip()

def processar_dados():
    try:
        # Garantir que a pasta de destino existe
        os.makedirs("dados", exist_ok=True)

        # Converter JSON para DataFrame
        print("Lendo arquivo JSON...")
        with open(noticia_json, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        df = pd.DataFrame(dados)

        # Limpa espaços nos nomes das colunas
        df.columns = df.columns.str.strip()
        
        # Aplica a limpeza na coluna de texto
        if 'texto' in df.columns:
            df['texto'] = df['texto'].apply(limpar_texto_noticia)
        
        # Reordenar colunas conforme desejado
        estrutura_desejada = ["id", "titulo", "texto", "data", "fonte"]
        # Filtrar apenas as colunas que realmente existem no JSON para evitar erro
        colunas_disponiveis = [col for col in estrutura_desejada if col in df.columns]
        df_final = df[colunas_disponiveis]

        
        # Exportamos primeiro o arquivo limpo (intermediário)
        df_final.to_csv(noticia_limpa, index=False, encoding='utf-8')
        
        # Exporta o arquivo final formatado
        df_final.to_csv(noticia_organizada, index=False, encoding='utf-8')
        print(f"Dados processados e limpos com sucesso!")
        print(f"Arquivo final gerado: {noticia_organizada}")
        print(f"Colunas processadas: {list(df_final.columns)}")
        print(df_final.head(3))

    except FileNotFoundError:
        print(f"Erro: O arquivo '{noticia_json}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# chama a função principal para processar os dados
if __name__ == "__main__":
    processar_dados()