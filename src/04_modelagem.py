import json
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
RAIZ = Path(__file__).resolve().parents[1]
PROC, FIG, RES, MOD = (RAIZ / 'data' / 'processed', RAIZ / 'figures', RAIZ / 'results', RAIZ / 'models')
for d in (FIG, RES, MOD):
    d.mkdir(exist_ok=True)
SEMENTE = 42
TEST_FRAC = 0.25
ATRIBUTOS = ['var_gasolina', 'var_diesel', 'var_etanol', 'var_gasolina_lag1', 'var_diesel_lag1', 'var_etanol_lag1', 'ipca_lag1', 'mes_calendario']
COR = {'base': '#8a8985', 'linear': '#2a78d6', 'arvore': '#eb6834', 'bom': '#1baf7a'}
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e3e2de', 'grid.linewidth': 0.6, 'legend.frameon': False, 'axes.titleweight': 'bold', 'savefig.dpi': 200})

def metricas(y, p):
    return {'MAE': mean_absolute_error(y, p), 'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'R2': r2_score(y, p)}

def main():
    b = pd.read_csv(PROC / 'base_modelagem.csv')
    b['mes_p'] = pd.PeriodIndex(b['mes'], freq='M')
    df = b.dropna(subset=ATRIBUTOS + ['ipca']).sort_values('mes_p').reset_index(drop=True)
    n = len(df)
    corte = int(n * (1 - TEST_FRAC))
    tr, te = (df.iloc[:corte], df.iloc[corte:])
    Xtr, ytr, Xte, yte = (tr[ATRIBUTOS], tr['ipca'], te[ATRIBUTOS], te['ipca'])
    print(f'Amostra util: {n} meses | treino {len(tr)} ({tr.mes.iloc[0]}..{tr.mes.iloc[-1]}) | teste {len(te)} ({te.mes.iloc[0]}..{te.mes.iloc[-1]})')
    cv = TimeSeriesSplit(n_splits=5)
    candidatos = {'Baseline: média do treino': (DummyRegressor(strategy='mean'), {}), 'Baseline: IPCA do mês anterior': (None, {}), 'Regressão Linear': (Pipeline([('sc', StandardScaler()), ('m', LinearRegression())]), {}), 'Ridge': (Pipeline([('sc', StandardScaler()), ('m', Ridge())]), {'m__alpha': [0.1, 1, 10, 30, 100]}), 'Random Forest': (RandomForestRegressor(random_state=SEMENTE, n_jobs=-1), {'n_estimators': [200], 'max_depth': [2, 3, 5], 'min_samples_leaf': [3, 8]}), 'Gradient Boosting': (GradientBoostingRegressor(random_state=SEMENTE), {'n_estimators': [100, 300], 'learning_rate': [0.03, 0.1], 'max_depth': [1, 2]})}
    linhas, ajustados, previsoes = ([], {}, {})
    for nome, (est, grade) in candidatos.items():
        if est is None:
            p_tr, p_te = (Xtr['ipca_lag1'].values, Xte['ipca_lag1'].values)
            cv_rmse = float(np.sqrt(mean_squared_error(ytr, p_tr)))
            melhor = None
        else:
            if grade:
                gs = GridSearchCV(est, grade, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).fit(Xtr, ytr)
                melhor, cv_rmse = (gs.best_estimator_, -gs.best_score_)
                params = gs.best_params_
            else:
                scores = []
                for i, (a, v) in enumerate(cv.split(Xtr)):
                    m = est.__class__(**est.get_params()) if not isinstance(est, Pipeline) else est
                    m.fit(Xtr.iloc[a], ytr.iloc[a])
                    scores.append(np.sqrt(mean_squared_error(ytr.iloc[v], m.predict(Xtr.iloc[v]))))
                cv_rmse, params = (float(np.mean(scores)), {})
                melhor = est.fit(Xtr, ytr)
            p_tr, p_te = (melhor.predict(Xtr), melhor.predict(Xte))
        m_te = metricas(yte, p_te)
        linhas.append({'modelo': nome, 'RMSE_cv_treino': cv_rmse, 'MAE_teste': m_te['MAE'], 'RMSE_teste': m_te['RMSE'], 'R2_teste': m_te['R2'], 'hiperparametros': str(params) if melhor is not None and grade else ''})
        ajustados[nome], previsoes[nome] = (melhor, p_te)
    tab = pd.DataFrame(linhas)
    tab.to_csv(RES / 'comparacao_modelos.csv', index=False)
    print(tab.round(3).to_string(index=False))
    elegiveis = tab[~tab['modelo'].str.startswith('Baseline')]
    escolhido = elegiveis.sort_values('RMSE_cv_treino').iloc[0]['modelo']
    modelo = ajustados[escolhido]
    print('Modelo escolhido (CV temporal):', escolhido)
    ordem = tab.sort_values('RMSE_teste', ascending=False)
    cores = [COR['base'] if m.startswith('Baseline') else COR['bom'] if m == escolhido else COR['linear'] for m in ordem['modelo']]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(ordem['modelo'], ordem['RMSE_teste'], color=cores, height=0.62)
    for y, v in enumerate(ordem['RMSE_teste']):
        ax.text(v + 0.005, y, f'{v:.3f}', va='center', color='#0b0b0b', fontsize=9)
    ax.set_xlabel('RMSE no conjunto de teste (pontos percentuais de IPCA) — menor é melhor')
    ax.grid(axis='y', visible=False)
    ax.set_title('Comparação dos modelos (cinza = baselines; verde = escolhido)')
    plt.tight_layout()
    plt.savefig(FIG / 'fig07_comparacao_modelos.png', bbox_inches='tight', facecolor='white')
    plt.close()
    datas = te['mes_p'].dt.to_timestamp()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(datas, yte.values, color='#0b0b0b', lw=2, marker='o', ms=4, label='IPCA observado')
    ax.plot(datas, previsoes[escolhido], color='#2a78d6', lw=2, marker='o', ms=4, label=f'Previsto ({escolhido})')
    ax.plot(datas, previsoes['Baseline: média do treino'], color='#8a8985', lw=1.5, ls='--', label='Baseline: média')
    ax.set_ylabel('IPCA mensal (%)')
    ax.set_ylim(-0.25, 1.6)
    ax.legend(ncol=3, loc='upper left')
    ax.set_title('Conjunto de teste: IPCA observado vs. previsto')
    plt.tight_layout()
    plt.savefig(FIG / 'fig08_previsto_observado.png', bbox_inches='tight', facecolor='white')
    plt.close()
    imp = permutation_importance(modelo, Xte, yte, n_repeats=50, random_state=SEMENTE, scoring='neg_root_mean_squared_error')
    imp_df = pd.DataFrame({'atributo': ATRIBUTOS, 'importancia': imp.importances_mean, 'desvio': imp.importances_std}).sort_values('importancia', ascending=False)
    imp_df.to_csv(RES / 'importancia_atributos.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(imp_df['atributo'][::-1], imp_df['importancia'][::-1], xerr=imp_df['desvio'][::-1], color='#2a78d6', height=0.6, error_kw={'ecolor': '#52514e', 'lw': 1})
    ax.set_xlabel('Aumento do RMSE ao embaralhar o atributo')
    ax.grid(axis='y', visible=False)
    ax.set_title(f'Importância por permutação — {escolhido}')
    plt.tight_layout()
    plt.savefig(FIG / 'fig09_importancia.png', bbox_inches='tight', facecolor='white')
    plt.close()
    limiar = float(np.percentile(ytr, 75))
    ctr, cte = ((ytr > limiar).astype(int), (yte > limiar).astype(int))
    print(f'Limiar IPCA alto = {limiar:.2f}% | positivos treino {ctr.mean():.0%} | teste {cte.mean():.0%}')
    clfs = {'Logística (sem balanceamento)': Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=1000))]), 'Logística (class_weight=balanced)': Pipeline([('sc', StandardScaler()), ('m', LogisticRegression(max_iter=1000, class_weight='balanced'))]), 'Random Forest (balanced_subsample)': RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3, class_weight='balanced_subsample', random_state=SEMENTE)}
    lc = []
    for nome, clf in clfs.items():
        clf.fit(Xtr, ctr)
        pr, pb = (clf.predict(Xte), clf.predict_proba(Xte)[:, 1])
        lc.append({'modelo': nome, 'precisao': precision_score(cte, pr, zero_division=0), 'recall': recall_score(cte, pr), 'F1': f1_score(cte, pr), 'ROC_AUC': roc_auc_score(cte, pb) if cte.nunique() > 1 else np.nan})
    pd.DataFrame(lc).to_csv(RES / 'comparacao_classificadores.csv', index=False)
    print(pd.DataFrame(lc).round(3).to_string(index=False))
    joblib.dump({'modelo': modelo, 'atributos': ATRIBUTOS, 'nome': escolhido}, MOD / 'modelo_final.joblib')
    out = {'n_meses_utilizaveis': n, 'n_treino': len(tr), 'n_teste': len(te), 'periodo_treino': [str(tr.mes.iloc[0]), str(tr.mes.iloc[-1])], 'periodo_teste': [str(te.mes.iloc[0]), str(te.mes.iloc[-1])], 'modelo_escolhido': escolhido, 'limiar_ipca_alto': limiar, 'metricas_teste_escolhido': metricas(yte, previsoes[escolhido]), 'metricas_teste_baseline_media': metricas(yte, previsoes['Baseline: média do treino']), 'metricas_teste_baseline_naive': metricas(yte, previsoes['Baseline: IPCA do mês anterior'])}
    (RES / 'metricas.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))
if __name__ == '__main__':
    main()
