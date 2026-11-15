from pathlib import Path
import numpy as np
import pandas as pd
RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / 'data' / 'processed'
PRODUTOS = {'GASOLINA': 'gasolina', 'DIESEL': 'diesel', 'ETANOL': 'etanol'}

def main():
    m = pd.read_csv(PROC / 'mensal_produto.csv')
    m['mes'] = pd.PeriodIndex(m['mes'], freq='M')
    precos = m.pivot(index='mes', columns='Produto', values='media')[list(PRODUTOS)].rename(columns=PRODUTOS)
    precos = precos.sort_index()
    consecutivo = pd.Series(precos.index.asi8, index=precos.index).diff() == 1
    var = precos.pct_change() * 100
    var = var.where(consecutivo, np.nan)
    var.columns = [f'var_{c}' for c in var.columns]
    ipca = pd.read_csv(PROC / 'ipca_mensal_1979-2026.csv', parse_dates=['data'])
    ipca['mes'] = ipca['data'].dt.to_period('M')
    ipca = ipca.set_index('mes')[['ipca_variacao_mensal']].rename(columns={'ipca_variacao_mensal': 'ipca'})
    base = precos.join(var).join(ipca, how='left').reset_index()
    base['mes'] = base['mes'].astype(str)
    base.to_csv(PROC / 'base_modelagem_bruta.csv', index=False)
    b = base.copy()
    b['mes_p'] = pd.PeriodIndex(b['mes'], freq='M')
    b = b.set_index('mes_p')
    cons = pd.Series(b.index.asi8, index=b.index).diff() == 1
    for c in ['var_gasolina', 'var_diesel', 'var_etanol', 'ipca']:
        b[f'{c}_lag1'] = b[c].shift(1).where(cons)
    b['mes_calendario'] = b.index.month
    b['ano'] = b.index.year
    b = b.reset_index(drop=True)
    b.to_csv(PROC / 'base_modelagem.csv', index=False)
    util = b.dropna(subset=['var_gasolina', 'var_diesel', 'var_etanol', 'ipca'])
    print(f'Meses com preco: {len(b)}  | com variacao e IPCA: {len(util)}  | com defasagem: {len(util.dropna())}')
if __name__ == '__main__':
    main()
