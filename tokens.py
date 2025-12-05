import pandas as pd
import spacy
import typer
from pathlib import Path
from tqdm import tqdm
import sys

app = typer.Typer()

# dicionario pra mapear o nome do idioma que ta no csv pro modelo do spacy
# tem que bater certinho com o que saiu do script anterior
modelos = {
    'Portugues': 'pt_core_news_sm',
    'Espanhol': 'es_core_news_sm',
    'Inglês': 'en_core_web_sm',
    # coloquei essas variações só pra garantir se vier diferente
    'Ingles': 'en_core_web_sm',
    'pt': 'pt_core_news_sm',
    'es': 'es_core_news_sm',
    'en': 'en_core_web_sm'
}

def carregar_modelos():
    # carrega os modelos do spacy pra memoria
    # tirei o parser e ner pra ficar mais rapido
    cache = {}
    print("carregando modelos...")
    
    for idioma, nome_modelo in modelos.items():
        # verifica se ja carregou pra nao carregar duas vezes a mesma coisa
        if nome_modelo not in cache.values():
            try:
                nlp = spacy.load(nome_modelo, disable=['parser', 'ner'])
                cache[idioma] = nlp
            except OSError:
                print(f"aviso: nao achei o modelo {nome_modelo}. tem que baixar antes")
    
    return cache

def pegar_lemas(texto, idioma, cache):
    # funcao que roda linha por linha pra limpar o texto
    
    # se nao tiver texto ou nao tiver modelo pro idioma, retorna vazio
    if pd.isna(texto) or idioma not in cache:
        return []
    
    # pega o nlp certo pro idioma
    nlp = cache[idioma]
    
    # corta o texto se for gigante pra nao travar tudo
    texto_seguro = str(texto)[:100000]
    
    # processa o texto
    doc = nlp(texto_seguro)
    
    # aqui acontece a limpeza
    # pega o lema , joga pra minusculo
    # e tira pontuacao e espacos vazios
    lista_limpa = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_punct and not token.is_space
    ]
    
    return lista_limpa

@app.command()
def main(
    input_file: str = "data/interim/brasil_lang.csv",
    output_file: str = "data/processed/brasil_tokens.parquet"
):
    print(f"lendo arquivo: {input_file}")
    
    path_in = Path(input_file)
    if not path_in.exists():
        print(f"erro: nao encontrei o arquivo {input_file}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(path_in)
    except Exception as e:
        print(f"deu erro pra ler o csv: {e}")
        sys.exit(1)

    # confere se tem a coluna de idioma que a gente precisa
    if 'idioma' not in df.columns:
        print("erro: ta faltando a coluna 'idioma'. roda o script de detectar idioma antes")
        sys.exit(1)

    # carrega os modelos antes de comecar o loop
    meus_modelos = carregar_modelos()
    
    # barra de progresso pra acompanhar
    tqdm.pandas(desc="processando")
    
    print(f"processando {len(df)} linhas...")
    
    # aplica a funcao em cada linha do dataframe
    df['lemas'] = df.progress_apply(
        lambda linha: pegar_lemas(linha.get('texto'), linha.get('idioma'), meus_modelos), 
        axis=1
    )
    
    # cria a pasta se nao existir
    path_out = Path(output_file)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    
    # organiza as colunas pra salvar
    cols_finais = ['texto', 'idioma', 'lemas']
    
    # se tiver essas colunas no original, mantem elas tambem
    cols_extras = ['id', 'data', 'pais', 'codigo legenda']
    for col in cols_extras:
        # gambiarra pra achar a coluna mesmo se tiver maiuscula/minuscula diferente
        achei = [c for c in df.columns if c.lower() == col.lower()]
        if achei:
            cols_finais.insert(0, achei[0])
            
    # salva em parquet que é melhor pra listas
    df[cols_finais].to_parquet(path_out, index=False)
    
    print("pronto!")
    print(f"arquivo salvo em: {output_file}")

if __name__ == "__main__":
    app()