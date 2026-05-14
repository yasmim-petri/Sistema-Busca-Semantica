# Sistema de Busca Semântica de Notícias com Embeddings

## Descrição

Este projeto realiza o processamento, limpeza e indexação semântica de notícias utilizando embeddings gerados por modelos de linguagem da biblioteca SentenceTransformers.

O sistema:

- Carrega notícias em formato JSON;
- Realiza limpeza textual;
- Gera embeddings vetoriais multilíngues;
- Salva os embeddings em arquivo `.pkl`;
- Permite consultas semânticas utilizando similaridade por cosseno.

Modelo utilizado:

`ibm-granite/granite-embedding-97m-multilingual-r2` foi escolhido por apresentar um bom equilíbrio entre:

- desempenho semântico;
- velocidade de processamento;
- baixo consumo de memória;
- leve exigência de hardware;
- suporte multilíngue.

Diferente de modelos maiores, que exigem GPUs mais robustas e maior quantidade de memória RAM, este modelo consegue gerar embeddings de forma eficiente até mesmo em máquinas mais simples ou ambientes gratuitos como o Google Colab. Sua arquitetura reduz o tempo de processamento durante a geração dos embeddings, tornando a busca semântica mais rápida e prática para aplicações.

---

# Estrutura do Projeto

```bash
projeto/
│
├── dados/
│   ├── noticias_brutas.json
│   ├── noticias_limpas.csv
│   └── embeddings_noticias.pkl
│
├── main.py
└── README.md
```

---

# Tecnologias Utilizadas

- Python 3.12 (Recomendada)
- Pandas
- NumPy
- SentenceTransformers
- Scikit-learn
- Pickle

---

# Instalação

Clone o repositório:

```bash
git clone <https://github.com/yasmim-petri/Sistema-Busca-Semantica.git>
cd <Sistema-Busca-Semantica>
```

Instale as dependências:

```bash
pip install pandas numpy sentence-transformers scikit-learn
```

---

# Formato do JSON de Entrada

O arquivo `noticias_brutas.json` deve possuir uma lista de objetos no seguinte formato:

```json
[
  {
    "id": 1,
    "titulo": "Banco Central altera taxa de juros",
    "texto": "O Banco Central anunciou novas mudanças...",
    "data": "2026-05-13",
    "fonte": "Portal Econômico"
  }
]
```

---

# Funcionamento do Sistema

## 1. Limpeza dos Dados

A função `limpar_texto_noticia()` realiza:

- Remoção de tags HTML;
- Conversão de entidades HTML;
- Remoção de espaços extras.

Exemplo:

```python
def limpar_texto_noticia(texto):
    if not isinstance(texto, str):
        return ""

    texto = html.unescape(texto)
    texto = re.sub(r'<[^>]+>', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)

    return texto.strip()
```

---

## 2. Processamento das Notícias

A função `processar_dados()`:

- Carrega o JSON;
- Cria um DataFrame;
- Limpa os textos;
- Salva os dados tratados em CSV.

Arquivo gerado:

```bash
dados/noticias_limpas.csv
```

---

## 3. Geração de Embeddings

Os embeddings são criados utilizando:

```python
model = SentenceTransformer(
    'ibm-granite/granite-embedding-97m-multilingual-r2'
)
```

Os vetores representam semanticamente o conteúdo das notícias.

Para melhorar a busca, o sistema concatena:

```python
titulo + texto
```

---

## 4. Salvamento dos Embeddings

Os dados são armazenados em:

```bash
dados/embeddings_noticias.pkl
```

Estrutura salva:

```python
{
    'embeddings': embeddings,
    'textos': textos,
    'titulos': titulos
}
```

---

# Busca Semântica

O sistema permite realizar consultas em linguagem natural.

Exemplo:

```python
query = "mudanças na taxa de juros"
```

O processo:

1. Gera embedding da consulta;
2. Calcula similaridade por cosseno;
3. Retorna as notícias mais relevantes.

---

# Exemplo de Resultado

```bash
Consultar: mudanças na taxa de juros

Score: 0.8731
ID: 15
Titulo: Banco Central anuncia aumento da Selic
Texto: O Banco Central decidiu elevar...
Fonte: Portal Econômico
```

---

# Métrica Utilizada

A similaridade entre os vetores é calculada usando:

```math
\cos(\theta)=\frac{A \cdot B}{||A|| ||B||}
```

Onde:

- `A` representa o embedding da consulta;
- `B` representa o embedding do documento.

---

# Principais Funcionalidades

- Limpeza automática de texto;
- Processamento de arquivos JSON;
- Geração de embeddings multilíngues;
- Busca semântica;
- Recuperação das notícias mais similares;
- Persistência de embeddings em arquivo.

---

# Execução

Execute o projeto com:

```bash
python main.py
```

---

# Avaliação Qualitativa dos Resultados

Os resultados obtidos demonstraram que o modelo consegue identificar relações semânticas entre a consulta e as notícias, mesmo quando não há correspondência exata de palavras.

Por exemplo, consultas relacionadas a:

```python
"mudanças na taxa de juros"
```

retornaram notícias contendo termos como:

- Selic;
- Banco Central;
- política monetária;
- inflação;
- economia.

Isso mostra que o sistema compreende o contexto semântico da busca, e não apenas a ocorrência literal das palavras.

Além disso:

- notícias mais relevantes apresentaram scores maiores;
- consultas genéricas retornaram conteúdos semanticamente próximos;
- a combinação de título + texto melhorou significativamente a precisão dos resultados.
