from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
RAIZ = Path(__file__).resolve().parents[1]
PROC, RES = (RAIZ / 'data' / 'processed', RAIZ / 'results')
PRODUTOS = {'GASOLINA': 'gasolina', 'DIESEL': 'diesel', 'ETANOL': 'etanol'}

def variacao(precos):
    precos = precos.sort_index()
    consecutivo = pd.Series(precos.index.asi8, index=precos.index).diff() == 1
    return (precos.pct_change() * 100).where(consecutivo, np.nan)

def main():
    uf = pd.read_csv(PROC / 'mensal_produto_uf.csv')
    uf = uf[uf['Produto'].isin(PRODUTOS)].copy()
    uf['mes'] = pd.PeriodIndex(uf['mes'], freq='M')
    uf['media_uf'] = uf['soma'] / uf['n']
    ipca = pd.read_csv(PROC / 'ipca_mensal_1979-2026.csv', parse_dates=['data'])
    ipca['mes'] = ipca['data'].dt.to_period('M')
    ipca = ipca.set_index('mes')['ipca_variacao_mensal']
    esquemas = {'media simples dos postos': uf.groupby(['mes', 'Produto']).apply(lambda d: d['soma'].sum() / d['n'].sum()).unstack('Produto'), 'media das medias estaduais (peso igual por UF)': uf.groupby(['mes', 'Produto'])['media_uf'].mean().unstack('Produto')}
    linhas = []
    for nome, precos in esquemas.items():
        var = variacao(precos[list(PRODUTOS)].rename(columns=PRODUTOS))
        base = var.join(ipca, how='inner').dropna()
        for p in PRODUTOS.values():
            r, pv = stats.pearsonr(base[p], base['ipca_variacao_mensal'])
            linhas.append({'ponderacao': nome, 'produto': p, 'n_meses': len(base), 'pearson_r': r, 'p_valor': pv})
    out = pd.DataFrame(linhas)
    out.to_csv(RES / 'sensibilidade_ponderacao_regional.csv', index=False)
    print(out.round(4).to_string(index=False))
if __name__ == '__main__':
    main()
