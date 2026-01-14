import pandas as pd
import typer
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
from bertopic import BERTopic
from sklearn.cluster import KMeans
from umap import UMAP

# Configurações do Backlog
HDB_MIN_CLUSTER_SIZE = 12
UMAP_N_NEIGHBORS = 15
KMEANS_N_CLUSTERS = 30 

app = typer.Typer()
tqdm.pandas()

def save_topic_results(df: pd.DataFrame, output_path: Path):
    """Salva o DataFrame de tópicos em CSV, removendo colunas pesadas."""
    cols_to_save = [c for c in df.columns if c not in ['embedding', 'texto']]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"   -> Dados salvos: {output_path.name}")

@app.command()
def main(
    input_file: str = typer.Argument("data/embeddings/embeddings.parquet", help="Caminho para o arquivo embeddings.parquet"),
    output_dir: str = typer.Option("results/topics/", "--output-dir", "-o", help="Pasta de saída"),
):
    print("--- Tarefa 3.2/3.3: Modelagem de Tópicos (BERTopic + HTML) ---")

    path_in = Path(input_file)
    if not path_in.exists():
        print(f"ERRO: Arquivo não encontrado: {input_file}")
        sys.exit(1)

    print(f"Lendo: {input_file}")
    df = pd.read_parquet(path_in)

    # --- 1. CORREÇÃO DE DADOS (Coluna 'pais') ---
    if 'pais' not in df.columns:
        if 'id' in df.columns:
            print("AVISO: Coluna 'pais' não encontrada. Criando a partir da coluna 'id'...")
            # Pega tudo antes do primeiro underscore '_' (Ex: 'Brasil_0001' -> 'Brasil')
            df['pais'] = df['id'].astype(str).apply(lambda x: x.split('_')[0] if '_' in x else 'Desconhecido')
        else:
            print("ERRO CRÍTICO: Faltam colunas 'pais' e 'id'. Impossível segmentar.")
            sys.exit(1)
    # -------------------------------------------

    if 'embedding' not in df.columns or 'texto' not in df.columns:
        print(f"ERRO: Faltam colunas obrigatórias. Encontrado: {list(df.columns)}")
        sys.exit(1)

    # 2. PREPARAÇÃO
    print("Preparando matriz de embeddings...")
    embeddings = np.stack(df['embedding'].values)
    texts = df['texto'].tolist()
    
    # 3. TREINAMENTO (KMeans para estabilidade)
    print(f"Iniciando treinamento Global (KMeans K={KMEANS_N_CLUSTERS})...")
    
    clustering_model = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=42, n_init='auto')
    umap_model = UMAP(n_neighbors=UMAP_N_NEIGHBORS, metric='cosine', random_state=42)
    
    model = BERTopic(
        hdbscan_model=clustering_model,
        umap_model=umap_model,
        verbose=True
    )

    # Treina e transforma
    topics, probs = model.fit_transform(texts, embeddings=embeddings)
    
    df['topic_id'] = topics
    df['topic_prob'] = probs
    
    # 4. SALVAMENTO E VISUALIZAÇÃO
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Salva modelo
    model.save(str(output_dir_path / "global_bertopic_model"))

    print("\nGerando resultados por país (CSV + HTML):")
    for pais, df_pais in df.groupby('pais'):
        pais_slug = pais.lower().replace(' ', '_')
        
        # A. Salvar CSV
        fname_csv = f"topics_{pais_slug}.csv"
        save_topic_results(df_pais, output_dir_path / fname_csv)
        
        # B. Gerar e Salvar HTML
        try:
            # Identifica quais tópicos existem neste país
            topics_in_country = df_pais['topic_id'].unique().tolist()
            if -1 in topics_in_country: topics_in_country.remove(-1) # Remove outliers
            
            if topics_in_country:
                # Gera o gráfico interativo apenas para os tópicos do país
                fig = model.visualize_topics(topics=topics_in_country)
                fname_html = f"topics_{pais_slug}.html"
                fig.write_html(str(output_dir_path / fname_html))
                print(f"   -> Visualização: {fname_html}")
        except Exception as e:
            print(f"   [!] Erro ao gerar HTML para {pais}: {e}")

    # 5. RESULTADOS GLOBAIS
    summary_path = output_dir_path / "global_topic_summary.csv"
    model.get_topic_info().to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    try:
        fig_global = model.visualize_topics()
        fig_global.write_html(str(output_dir_path / "global_topics.html"))
        print(f"\n✅ Visualização Global salva em: global_topics.html")
    except:
        pass

    print(f"\nTAREFA 3.2/3.3 CONCLUÍDA! Verifique a pasta: {output_dir}")

if __name__ == "__main__":
    app()