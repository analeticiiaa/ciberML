import pandas as pd
import typer
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
from bertopic import BERTopic
from umap import UMAP


HDB_MIN_CLUSTER_SIZE = 12
UMAP_N_NEIGHBORS = 15

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
    print("--- Tarefa 3.2/3.3: Modelagem de Tópicos (BERTopic Final) ---")

    path_in = Path(input_file)
    if not path_in.exists():
        print(f"ERRO: Arquivo não encontrado: {input_file}")
        sys.exit(1)

    print(f"Lendo: {input_file}")
    df = pd.read_parquet(path_in)

    
    if 'pais' not in df.columns:
        if 'id' in df.columns:
            print("AVISO: Coluna 'pais' não encontrada. Criando a partir da coluna 'id'...")
            # Ex: 'Brasil_0001' -> pega 'Brasil'
            df['pais'] = df['id'].astype(str).apply(lambda x: x.split('_')[0] if '_' in x else 'Desconhecido')
        else:
            print("ERRO CRÍTICO: Faltam colunas 'pais' e 'id'. Impossível segmentar por país.")
            sys.exit(1)
  
    if 'embedding' not in df.columns or 'texto' not in df.columns:
        print(f"ERRO: Faltam colunas obrigatórias 'embedding' ou 'texto'.")
        sys.exit(1)

    # 2. PREPARAÇÃO
    print("Preparando matriz de embeddings...")
    
    embeddings = np.stack(df['embedding'].values)
    texts = df['texto'].tolist()
    
    # 3. CONFIGURAÇÃO E TREINAMENTO
    print(f"Configurando UMAP (n_neighbors={UMAP_N_NEIGHBORS})...")
    
    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS, 
        n_components=5, 
        min_dist=0.0, 
        metric='cosine', 
        random_state=42
    )
    
    print(f"Iniciando treinamento Global (HDBSCAN min_cluster={HDB_MIN_CLUSTER_SIZE})...")
    model = BERTopic(
        min_topic_size=HDB_MIN_CLUSTER_SIZE,
        umap_model=umap_model, 
        verbose=True
    )

    
    topics, probs = model.fit_transform(texts, embeddings=embeddings)
    
 
    df['topic_id'] = topics
    df['topic_prob'] = probs
    
    # 4. SALVAMENTO E VISUALIZAÇÃO
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    model.save(str(output_dir_path / "global_bertopic_model"))

    print("\nGerando resultados por país (Tarefa 3.2):")
    for pais, df_pais in df.groupby('pais'):
        pais_slug = pais.lower().replace(' ', '_')
        
        # A. Salvar CSV (Dados)
        fname_csv = f"topics_{pais_slug}.csv"
        save_topic_results(df_pais, output_dir_path / fname_csv)
        
        # B. Gerar Visualização (HTML)
        try:
            topics_in_country = df_pais['topic_id'].unique().tolist()
            # Remove ruído (-1) para focar nos tópicos reais
            if -1 in topics_in_country: topics_in_country.remove(-1)
            
            if topics_in_country:
                # Usa BarChart que é mais robusto para poucos tópicos
                top_n = min(len(topics_in_country), 10)
                fig = model.visualize_barchart(topics=topics_in_country[:top_n], top_n_topics=top_n)
                
                fname_html = f"topics_{pais_slug}.html"
                fig.write_html(str(output_dir_path / fname_html))
                print(f"   -> Visualização: {fname_html}")
            else:
                print(f"   [i] {pais}: Nenhum tópico consistente encontrado (apenas ruído).")
        except Exception as e:
            print(f"   [!] Erro ao gerar HTML para {pais}: {e}")

   
    summary_path = output_dir_path / "global_topic_summary.csv"
    model.get_topic_info().to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    
    try:
        model.visualize_topics().write_html(str(output_dir_path / "global_topics_map.html"))
        model.visualize_barchart(top_n_topics=20).write_html(str(output_dir_path / "global_topics_bar.html"))
        print(f"\n✅ Visualizações Globais salvas em: {output_dir}")
    except Exception as e:
        print(f"Aviso: Não foi possível gerar visualização global: {e}")

    print(f"\nTAREFA 3.2/3.3 CONCLUÍDA COM SUCESSO!")

if __name__ == "__main__":
    app()