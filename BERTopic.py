import pandas as pd
import typer
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
from bertopic import BERTopic

# Parâmetros otimizados do Backlog (Fonte 263-267)
HDB_MIN_CLUSTER_SIZE = 12
UMAP_N_NEIGHBORS = 15

app = typer.Typer()
tqdm.pandas()

def save_topic_results(df: pd.DataFrame, output_path: Path):
    # Salva apenas colunas essenciais para análise de tópico
    cols_to_save = [c for c in df.columns if c not in ['embedding', 'texto']]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Resultados de Tópicos salvos em: {output_path}")


@app.command()
def main(
    input_file: str = typer.Argument("data/embeddings/embeddings.parquet", help="Caminho para o arquivo embeddings.parquet"),
    output_dir: str = typer.Option("results/topics/", "--output-dir", "-o", help="Pasta de saída para relatórios e CSVs"),
):
    print("--- Tarefa 3.2/3.3: Modelagem de Tópicos (BERTopic) ---")

    path_in = Path(input_file)
    if not path_in.exists():
        print(f"ERRO: Arquivo de Embeddings não encontrado: {input_file}")
        sys.exit(1)

    print(f"Lendo corpus de embeddings: {input_file}")
    df = pd.read_parquet(path_in)

    if 'embedding' not in df.columns or 'texto' not in df.columns or 'pais' not in df.columns:
        print("ERRO: Colunas 'embedding', 'texto' ou 'pais' não encontradas. Verifique o input.")
        sys.exit(1)

    # 1. PREPARAÇÃO
    # O BERTopic espera a coluna embedding como uma matriz numpy 2D
    # (Transforma a Series de listas/arrays em uma matriz 2D)
    embeddings = np.stack(df['embedding'].values)
    texts = df['texto'].tolist()
    
    # 2. TREINAMENTO DO MODELO GLOBAL (Para uso nas Tarefas 3.2/3.3)
    print("Iniciando o treinamento do modelo BERTopic (Global)...")
    
    # Configurações do modelo BERTopic com base no backlog (Fonte 263-267)
    model = BERTopic(
        min_topic_size=HDB_MIN_CLUSTER_SIZE,
        umap_args={"n_neighbors": UMAP_N_NEIGHBORS},
        verbose=True 
    )

    # Treina o modelo e transforma os dados
    topics, probs = model.fit_transform(texts, embeddings=embeddings)
    
    df['topic_id'] = topics
    df['topic_prob'] = probs
    
    # 3. ANÁLISE DE RESULTADOS E SALVAMENTO POR PAÍS (Tarefa 3.2)
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Salva o modelo treinado (para reuso nas tarefas de visualização)
    model_output_path = output_dir_path / "global_bertopic_model"
    model.save(str(model_output_path))
    print(f"Modelo BERTopic salvo em: {model_output_path}")

    # Salva resultados de tópico para CADA PAÍS, segmentando pelo 'pais'
    print("\nGerando CSVs de resultados por país (Tarefa 3.2)...")
    for pais, df_pais in df.groupby('pais'):
        # Cria um arquivo CSV por país com o Tópico ID
        output_file_name = f"topics_{pais.lower().replace(' ', '_')}.csv"
        output_path = output_dir_path / output_file_name
        save_topic_results(df_pais, output_path)

    # Gera o relatório de tópicos (sumário global)
    report_path = output_dir_path / "global_topic_summary.csv"
    topic_summary = model.get_topic_info()
    topic_summary.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ Relatório de Tópicos Global salvo em: {report_path}")

if __name__ == "__main__":
    app()