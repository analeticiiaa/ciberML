import pandas as pd
import typer
from pathlib import Path
import yake

app = typer.Typer()

@app.command()
def main(
    input_file: str = "data/processed/mocambique_tokens.parquet",
    output_file: str = "results/keywords_mocambique.csv",
    top_n: int = 20
):
    print(f"Lendo arquivo: {input_file}")

    path_in = Path(input_file)
    if not path_in.exists():
        print(f"Erro: arquivo {input_file} não existe.")
        raise typer.Exit(code=1)

    df = pd.read_parquet(path_in)

    df["lemas"] = df["lemas"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else x)

    corpus = " ".join([" ".join(lemas) for lemas in df["lemas"] if isinstance(lemas, list)])

    print("Tamanho do corpus (tokens):", len(corpus.split()))

    #Formação principal do YAKE
    kw_extractor = yake.KeywordExtractor(
        lan="pt",
        n=1,
        top=top_n,
        dedupLim=0.9,
        features=None
    )

    keywords = kw_extractor.extract_keywords(corpus)

    result_df = pd.DataFrame(keywords, columns=["keyword", "score"])

    #Salvar tudo
    path_out = Path(output_file)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(path_out, index=False, encoding="utf-8-sig")

    print(f"\n✅ Keywords salvas em: {output_file}")
    print(result_df)


if __name__ == "__main__":
    app()
