import pandas as pd
import langid
import typer
from pathlib import Path
import sys

app = typer.Typer()

# configuracoes simples do projeto
min_chars = 20 
supported_languages = ['pt', 'es', 'en']

nome_idiomas = {
    'pt': 'Portugues',
    'es': 'Espanhol',
    'en': 'Ingles'
}

def detect_language(text: str) -> str:
    # se o texto estiver vazio, a gente ja descarta
    if pd.isna(text) or text is None:
        return "desconhecido"
    
    # remove espacos em branco sobrando no comeco e no fim
    text_str = str(text).strip()
    
    # se o texto for muito curtinho, ignoramos pra evitar erro
    if len(text_str) < min_chars:
        return "desconhecido"
    
    try:
        # aqui o langid tenta adivinhar qual é o idioma
        lang_code, _ = langid.classify(text_str)
        
        # aqui a gente traduz a sigla para o nome (ex:'pt' --> 'portugues')
        return nome_idiomas.get(lang_code, lang_code) 
    except Exception:
        # se der qualquer problema estranho, marcamos como erro
        return "erro"

@app.command()
def main(
    input_file: str = "brasil.csv",
    output_file: str = "data/interim/brasil_lang_completo.csv"
):
    print(f"Iniciando a leitura do arquivo: {input_file}")
    
    # avisa a biblioteca pra focar so nos nossos idiomas
    langid.set_languages(supported_languages)

    # verifica se o arquivo existe mesmo
    path_in = Path(input_file)
    if not path_in.exists():
        print(f"Erro: nao encontrei o arquivo {input_file}")
        sys.exit(1)

    try:
        # tenta ler o csv usando ponto e virgula como separador
        df = pd.read_csv(path_in, sep=';')
    except Exception as e:
        print(f"Erro ao tentar abrir o csv: {e}")
        sys.exit(1)

    # -------------------------------------------------------------
    # CORREÇÃO CRÍTICA: Trata 'text' e padroniza para 'texto'
    # -------------------------------------------------------------
    
    # transforma os nomes das colunas pra minusculo pra evitar confusao
    df.columns = [c.lower() for c in df.columns]

    # Verifica o nome correto da coluna e renomeia para 'texto'
    if 'texto' in df.columns:
        pass # A coluna já está no padrão
    elif 'text' in df.columns:
        # Renomeia 'text' para 'texto', seguindo o padrão do seu pipeline
        df.rename(columns={'text': 'texto'}, inplace=True)
        print("Ajuste: Coluna 'text' renomeada para 'texto'.")
    else:
        print(f"Erro: a coluna de conteúdo ('texto' ou 'text') não existe nesse arquivo.")
        sys.exit(1)
        
    # -------------------------------------------------------------
    
    print("Identificando os idiomas agora...")
    
    # aplica a nossa funcao linha por linha
    df['idioma'] = df['texto'].apply(detect_language)

    print("\n--- Resultado final ---")
    print(df['idioma'].value_counts())
    
    # cria a pasta de saida se ela ainda nao existir
    path_out = Path(output_file)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    
    # salva o arquivo novo pronto pra usar
    df.to_csv(path_out, index=False)
    
    print(f"\nTudo certo! arquivo salvo em: {output_file}")

if __name__ == "__main__":
    app()