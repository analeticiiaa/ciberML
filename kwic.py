import pandas as pd
import typer
from pathlib import Path
from tqdm import tqdm
import unicodedata 

app = typer.Typer()

## Função para normalizar as palavras, facilitando a comparação
def normalize(w):
    if not isinstance(w, str):
        return ""
    w = w.strip().lower()
    w = ''.join(c for c in unicodedata.normalize('NFD', w)
                if unicodedata.category(c) != 'Mn')
    return w

#formatação do KWIC no csv
def kwic_for_term(df, term, window=5):
    resultados = []
    term_norm = normalize(term)

    for idx, row in df.iterrows():
        lemas = row.get("lemas", [])

        if not isinstance(lemas, list):
            continue

        for i, lemma in enumerate(lemas):
            if normalize(lemma) == term_norm:

                left = " ".join(lemas[max(0, i - window):i])
                right = " ".join(lemas[i+1:i + window + 1])

                resultados.append({
                    "termo": term,
                    "linha": idx,
                    "before": left,
                    "keyword": lemma,
                    "after": right,
                    "texto_original": row.get("texto", ""),
                    "pais": row.get("pais", ""),
                    "idioma": row.get("idioma", "")
                })

    return pd.DataFrame(resultados)


def sample_kwic(df, n=10):
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=42)


#Criação do comando principal

@app.command()
def main(
    input_file: str = "data/processed/inglaterra_tokens.parquet",
    output_file: str = "results/kwic_inglaterra.csv",
    termos: list[str] = typer.Argument(...),
    window: int = 5
):

    print(f"Lendo arquivo: {input_file}")

    path_in = Path(input_file)
    if not path_in.exists():
        print(f"Erro: arquivo {input_file} não encontrado.")
        raise typer.Exit(code=1)

    df = pd.read_parquet(path_in)

    # Fazer com que todos os lemas sejam listas, do contrario voltara um documento vazio
    df['lemas'] = df['lemas'].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    # Lista de termos vem corretamente como lista
    lista_termos = [normalize(t) for t in termos]

    todos_kwics = []

    print("Gerando KWICs...")
    for termo in tqdm(lista_termos):
        df_kwic = kwic_for_term(df, termo, window=window)
        df_kwic_sample = sample_kwic(df_kwic, n=10)
        todos_kwics.append(df_kwic_sample)

    final_df = pd.concat(todos_kwics, ignore_index=True)

    path_out = Path(output_file)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(path_out, index=False, encoding="utf-8-sig")

    print(f"\n✅ KWIC salvo em: {output_file}")
    print(f"Total de linhas: {len(final_df)}")


if __name__ == "__main__":
    app()
