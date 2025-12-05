from sentence_transformers import SentenceTransformer
import pandas as pd
import typer
import sys 
from pathlib import Path
from tqdm import tqdm

app = typer.Typer()

@app.command()
def main(
    input_file: str = "data/processed/espanha_tokens.parquet",
    output_file: str = "data/embeddings/embeddings.parquet",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
):
    print(f"lendo arquivo: {input_file}")
    
    # confere se o arquivo de input existe
    path_in = Path(input_file)
    if not path_in.exists():
        print(f"erro: nao encontrei o arquivo {input_file}")
        sys.exit(1)
        
    try:
        df = pd.read_parquet(path_in)
    except Exception as e:
        print(f"deu erro pra ler o arquivo parquet: {e}")
        sys.exit(1)
    
    # confere se a coluna texto existe
    if 'texto' not in df.columns:
        print("erro: coluna 'texto' nao encontrada no arquivo de entrada")
        sys.exit(1)
    
    # carrega o modelo sentence transformer
    model = SentenceTransformer(model_name)
    
    # barra de progresso pra acompanhar
    tqdm.pandas(desc="processando")
    print(f"processando {len(df)} linhas...")
    
    # transforma a coluna 'texto' em uma lista
    embeddings = model.encode(df['texto'].tolist())

    # 
    df['embedding'] = list(embeddings)
    
    # cria a pasta se nao existir
    path_out = Path(output_file)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_file)
    
    print("pronto!")
    print(f"arquivo salvo em: {output_file}")

if __name__ == "__main__":
    app()