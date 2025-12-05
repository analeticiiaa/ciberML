import pandas as pd
import spacy
import typer
from pathlib import Path
from tqdm import tqdm
import sys

app = typer.Typer()
tqdm.pandas()

# 1. Mapeamento de Idioma para Modelos SpaCy
modelos = {
    'pt': 'pt_core_news_sm',
    'es': 'es_core_news_sm',
    'en': 'en_core_web_sm'
}

# Tipos de entidades de interesse (PER, ORG, LOCAL)
ENTITIES_OF_INTEREST = ['PER', 'ORG', 'LOC', 'PERSON', 'GPE']

def carregar_modelos():
    """Carrega os modelos do spaCy, garantindo que o sentencizer esteja ativo para extração de contexto."""
    cache = {}
    print("carregando modelos para NER...")
    
    for lang_code, nome_modelo in modelos.items():
        if nome_modelo not in cache.values():
            try:
                # Carrega o modelo, excluindo componentes pesados e desnecessários
                nlp = spacy.load(nome_modelo, exclude=['parser', 'tagger', 'lemmatizer'])
                
                # -----------------------------
                # CORREÇÃO CRÍTICA (Erro E030)
                # Garante que o separador de frases esteja na pipeline para que ent.sent.text funcione.
                if "sentencizer" not in nlp.pipe_names:
                    # Adiciona o sentencizer ANTES do NER
                    nlp.add_pipe("sentencizer", before="ner")
                # -----------------------------
                    
                cache[lang_code] = nlp
            except OSError:
                print(f"aviso: Modelo {nome_modelo} não encontrado. Certifique-se de que foi baixado.")
                sys.exit(1)
    
    return cache

def extrair_entidades(texto, idioma, cache):
    """Processa o texto e extrai entidades relevantes."""
    
    # Normaliza a coluna 'idioma' para códigos ISO ('Portugues' -> 'pt')
    idioma_code = idioma.lower().strip()
    idioma_code = idioma_code if idioma_code in modelos else 'pt' # Fallback
    
    if pd.isna(texto) or idioma_code not in cache:
        return []
    
    nlp = cache[idioma_code]
    doc = nlp(str(texto)[:100000]) # Limita o texto para segurança

    entidades_encontradas = []
    for ent in doc.ents:
        # Verifica se o tipo de entidade é relevante (Pessoa, Organização, Local)
        if ent.label_ in ENTITIES_OF_INTEREST:
            entidades_encontradas.append({
                "texto_entidade": ent.text,
                "tipo_entidade": ent.label_,
                # ent.sent.text agora funciona!
                "contexto": ent.sent.text 
            })
            
    return entidades_encontradas

@app.command()
def main(
    # Argumento Posicional Obrigatório
    input_file: str = typer.Argument(..., help="Caminho para o arquivo de entrada (ex: data/interim/brasil_lang.csv)"),
    # Opção para Output
    output_file: str = typer.Option(None, "--output", "-o", help="Caminho para o arquivo CSV de saída (ex: results/ner_brasil.csv)")
):
    print(f"Lendo arquivo: {input_file}")
    
    path_in = Path(input_file)
    if not path_in.exists():
        print(f"erro: nao encontrei o arquivo {input_file}")
        sys.exit(1)
        
    try:
        df = pd.read_csv(path_in)
    except Exception as e:
        print(f"deu erro pra ler o csv: {e}")
        sys.exit(1)

    if 'texto' not in df.columns or 'idioma' not in df.columns:
        print("erro: ta faltando a coluna 'texto' ou 'idioma'.")
        sys.exit(1)

    # --- CORREÇÃO DE DADOS: Normaliza a coluna 'idioma' (Se vier como 'Portugues') ---
    df['idioma'] = df['idioma'].astype(str).str.lower().str.replace('portugues', 'pt')
    df['idioma'] = df['idioma'].str.replace('ingles', 'en')
    df['idioma'] = df['idioma'].str.replace('espanhol', 'es')
    # ---------------------------------------------------------------------------------

    meus_modelos = carregar_modelos()
    
    print(f"processando {len(df)} linhas para NER...")
    df['entidades_raw'] = df.progress_apply(
        lambda linha: extrair_entidades(linha.get('texto'), linha.get('idioma'), meus_modelos), 
        axis=1
    )
    
    # -------------------------------------------------------------
    # EXPANDIR O DATAFRAME (Explode)
    # -------------------------------------------------------------
    
    df_entidades = df.explode('entidades_raw')
    df_entidades = df_entidades.dropna(subset=['entidades_raw'])
    
    df_final = pd.concat([df_entidades.drop('entidades_raw', axis=1), 
                          df_entidades['entidades_raw'].apply(pd.Series)], axis=1)

    # Determinação do caminho de saída
    if output_file is None:
        file_name_stem = path_in.stem.replace('_lang', '_ner')
        output_file = f"results/{file_name_stem}.csv"
        
    path_out = Path(output_file)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    
    cols_finais = [c for c in df_final.columns if c in ['texto_entidade', 'tipo_entidade', 'contexto', 'idioma', 'pais']]
    df_final[cols_finais].to_csv(path_out, index=False, encoding="utf-8-sig")
    
    print("pronto!")
    print(f"Arquivo de Entidades (NER) salvo em: {output_file}")
    print(f"Total de entidades extraídas: {len(df_final)}")


if __name__ == "__main__":
    app()
    