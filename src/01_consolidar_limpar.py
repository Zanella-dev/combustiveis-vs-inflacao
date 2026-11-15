import argparse
import io
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
COLUNAS = ['Regiao - Sigla', 'Estado - Sigla', 'Municipio', 'Revenda', 'CNPJ da Revenda', 'Nome da Rua', 'Numero Rua', 'Complemento', 'Bairro', 'Cep', 'Produto', 'Data da Coleta', 'Valor de Venda', 'Valor de Compra', 'Unidade de Medida', 'Bandeira']
FATOR_OUTLIER = 3.0

def decodificar(binario: bytes) -> str:
    try:
        return binario.decode('utf-8-sig')
    except UnicodeDecodeError:
        return binario.decode('latin-1')

def eh_base_anp(texto: str) -> bool:
    primeira_linha = texto.split('\n', 1)[0]
    return 'Valor de Venda' in primeira_linha and 'Produto' in primeira_linha

def iterar_arquivos(entrada: Path):
    for caminho in sorted(entrada.glob('*')):
        if caminho.suffix.lower() == '.zip':
            with zipfile.ZipFile(caminho) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith('.csv'):
                        texto = decodificar(z.read(info))
                        if eh_base_anp(texto):
                            yield (info.filename, texto)
        elif caminho.suffix.lower() == '.csv':
            texto = decodificar(caminho.read_bytes())
            if eh_base_anp(texto):
                yield (caminho.name, texto)

def semestre_do_nome(nome: str) -> str:
    m = re.search('(20\\d\\d)[-.](0[12])', nome)
    return f'{m.group(1)}-S{m.group(2)[-1]}' if m else nome

def limpar(df: pd.DataFrame):
    c = {'linhas_brutas': len(df)}
    df = df.copy()
    df.columns = COLUNAS[:len(df.columns)]
    df['Data da Coleta'] = pd.to_datetime(df['Data da Coleta'], dayfirst=True, errors='coerce')
    for col in ('Valor de Venda', 'Valor de Compra'):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    c['compra_nula_pct'] = round(100 * df['Valor de Compra'].isna().mean(), 2)
    antes = len(df)
    df = df.dropna(subset=['Data da Coleta', 'Valor de Venda', 'Produto'])
    c['removidas_nulos'] = antes - len(df)
    antes = len(df)
    df = df[df['Valor de Venda'] > 0]
    c['removidas_preco_nao_positivo'] = antes - len(df)
    antes = len(df)
    df = df.drop_duplicates(subset=['CNPJ da Revenda', 'Produto', 'Data da Coleta', 'Valor de Venda'])
    c['removidas_duplicatas'] = antes - len(df)
    df['mes'] = df['Data da Coleta'].dt.to_period('M')
    mediana = df.groupby(['Produto', 'mes'])['Valor de Venda'].transform('median')
    razao = df['Valor de Venda'] / mediana
    ok = (razao >= 1 / FATOR_OUTLIER) & (razao <= FATOR_OUTLIER)
    c['removidas_outliers'] = int((~ok).sum())
    df = df[ok]
    c['linhas_finais'] = len(df)
    return (df, c)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--entrada', required=True, type=Path)
    ap.add_argument('--saida', type=Path, default=Path(__file__).resolve().parents[1] / 'data' / 'processed')
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    qualidade, por_uf, nacional, balanco = ([], [], [], [])
    for nome, texto in iterar_arquivos(args.entrada):
        bruto = pd.read_csv(io.StringIO(texto), sep=';', dtype=str)
        df, cont = limpar(bruto)
        cont['arquivo'] = nome
        cont['semestre'] = semestre_do_nome(nome)
        qualidade.append(cont)
        g = df.groupby(['Produto', 'mes', 'Estado - Sigla'])['Valor de Venda']
        por_uf.append(g.agg(soma='sum', n='count', mediana='median').reset_index())
        gn = df.groupby(['Produto', 'mes'])['Valor de Venda']
        nacional.append(gn.agg(media='mean', mediana='median', desvio='std', n='count').reset_index())
        df['ano'] = df['Data da Coleta'].dt.year
        balanco.append(df.groupby(['Produto', 'Regiao - Sigla', 'ano']).size().rename('n').reset_index())
        print(f"{cont['semestre']}: {cont['linhas_brutas']:>9,} -> {cont['linhas_finais']:>9,} linhas", flush=True)
    pd.DataFrame(qualidade).sort_values('semestre').to_csv(args.saida / 'qualidade_por_arquivo.csv', index=False)
    pd.concat(por_uf).assign(mes=lambda d: d['mes'].astype(str)).to_csv(args.saida / 'mensal_produto_uf.csv', index=False)
    pd.concat(nacional).assign(mes=lambda d: d['mes'].astype(str)).sort_values(['Produto', 'mes']).to_csv(args.saida / 'mensal_produto.csv', index=False)
    pd.concat(balanco).groupby(['Produto', 'Regiao - Sigla', 'ano'], as_index=False)['n'].sum().to_csv(args.saida / 'desbalanceamento.csv', index=False)
    print('Concluido.')
if __name__ == '__main__':
    main()
