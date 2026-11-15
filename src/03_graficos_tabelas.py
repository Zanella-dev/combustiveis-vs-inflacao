from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
RAIZ = Path(__file__).resolve().parents[1]
PROC, FIG, RES = (RAIZ / 'data' / 'processed', RAIZ / 'figures', RAIZ / 'results')
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
COR = {'gasolina': '#2a78d6', 'diesel': '#eb6834', 'etanol': '#1baf7a'}
TINTA, TINTA2, GRADE = ('#0b0b0b', '#52514e', '#e3e2de')
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 10, 'axes.edgecolor': GRADE, 'axes.labelcolor': TINTA2, 'xtick.color': TINTA2, 'ytick.color': TINTA2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRADE, 'grid.linewidth': 0.6, 'axes.titleweight': 'bold', 'axes.titlesize': 11, 'axes.titlecolor': TINTA, 'legend.frameon': False, 'figure.dpi': 100, 'savefig.dpi': 200})

def salvar(nome):
    plt.tight_layout()
    plt.savefig(FIG / nome, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    base = pd.read_csv(PROC / 'base_modelagem.csv')
    base['mes_p'] = pd.PeriodIndex(base['mes'], freq='M')
    base['data'] = base['mes_p'].dt.to_timestamp()
    prod = ['gasolina', 'diesel', 'etanol']
    cheio = base.set_index('data')[prod].asfreq('MS')
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for p in prod:
        ax.plot(cheio.index, cheio[p], color=COR[p], lw=2, label=p.capitalize())
    ax.set_ylabel('Preço médio de revenda (R$/L)')
    ax.legend(ncol=3, loc='upper left')
    ax.set_title('Preço médio mensal dos combustíveis no Brasil (lacunas = meses sem dados na base)')
    salvar('fig01_precos_mensais.png')
    disp = base.assign(ano=base['data'].dt.year, m=base['data'].dt.month).pivot_table(index='ano', columns='m', values='gasolina', aggfunc='count').reindex(range(2004, 2027)).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5.2))
    sns.heatmap(disp, cmap=['#f0efec', '#2a78d6'], cbar=False, linewidths=1, linecolor='white', ax=ax)
    ax.set_xlabel('Mês')
    ax.set_ylabel('Ano')
    ax.grid(False)
    ax.set_title('Disponibilidade da base ANP (azul = mês com dados)')
    salvar('fig02_disponibilidade.png')
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ipca = base.dropna(subset=['ipca'])
    ax.bar(ipca['data'], ipca['ipca'], width=25, color='#2a78d6')
    ax.set_ylabel('IPCA mensal (%)')
    ax.set_title('Variação mensal do IPCA nos meses cobertos pela base')
    salvar('fig03_ipca_mensal.png')
    util = base.dropna(subset=['var_gasolina', 'var_diesel', 'var_etanol', 'ipca']).copy()
    linhas = []
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, p in zip(axs, prod):
        x, y = (util[f'var_{p}'], util['ipca'])
        r, pv = stats.pearsonr(x, y)
        rho, _ = stats.spearmanr(x, y)
        ax.scatter(x, y, s=16, color=COR[p], alpha=0.75, edgecolor='white', linewidth=0.5)
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a * xs + b, color=TINTA2, lw=1.5)
        ax.set_title(f'{p.capitalize()}  (r = {r:.2f})')
        ax.set_xlabel('Variação mensal do preço (%)')
        linhas.append({'produto': p, 'n_meses': len(util), 'pearson_r': r, 'p_valor': pv, 'spearman_rho': rho, 'inclinacao': a})
    axs[0].set_ylabel('IPCA mensal (%)')
    salvar('fig04_dispersao.png')
    pd.DataFrame(linhas).to_csv(RES / 'correlacoes.csv', index=False)
    cols = ['var_gasolina', 'var_diesel', 'var_etanol', 'ipca', 'ipca_lag1', 'var_gasolina_lag1', 'var_diesel_lag1', 'var_etanol_lag1']
    corr = base[cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=sns.diverging_palette(220, 20, as_cmap=True), center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Matriz de correlação de Pearson entre os atributos')
    salvar('fig05_correlacao.png')
    corr.to_csv(RES / 'matriz_correlacao.csv')
    d = pd.read_csv(PROC / 'desbalanceamento.csv')
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    por_prod = d.groupby('Produto')['n'].sum().sort_values()
    axs[0].barh(por_prod.index, por_prod.values / 1000000.0, color='#2a78d6', height=0.6)
    axs[0].set_xlabel('Observações (milhões)')
    axs[0].set_title('Por produto')
    axs[0].grid(axis='y', visible=False)
    por_reg = d.groupby('Regiao - Sigla')['n'].sum().sort_values()
    axs[1].barh(por_reg.index, por_reg.values / 1000000.0, color='#1baf7a', height=0.6)
    axs[1].set_xlabel('Observações (milhões)')
    axs[1].set_title('Por região')
    axs[1].grid(axis='y', visible=False)
    salvar('fig06_desbalanceamento.png')
    pd.DataFrame({'produto': por_prod.index, 'n': por_prod.values, 'pct': 100 * por_prod.values / por_prod.sum()}).to_csv(RES / 'balanco_por_produto.csv', index=False)
    pd.DataFrame({'regiao': por_reg.index, 'n': por_reg.values, 'pct': 100 * por_reg.values / por_reg.sum()}).to_csv(RES / 'balanco_por_regiao.csv', index=False)
    desc = base[prod + ['var_gasolina', 'var_diesel', 'var_etanol', 'ipca']].describe().T
    desc.to_csv(RES / 'estatisticas_descritivas.csv')
    anual = base.assign(ano=base['data'].dt.year).groupby('ano')[prod].mean().round(3)
    anual.to_csv(RES / 'preco_medio_anual.csv')
    print(desc.round(3))
if __name__ == '__main__':
    main()
