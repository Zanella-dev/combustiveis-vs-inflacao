# Combustíveis vs. Inflação: mineração de dados com CRISP-DM

Trabalho final da disciplina de **Mineração de Dados** (grupo 6).

**Autores:** Kauã Zanela e Rafael Pereira.

**Pergunta de pesquisa:** a variação mensal dos preços de gasolina, diesel e etanol ajuda a explicar o IPCA mensal?

O artigo completo está em [`docs/artigo_combustiveis_inflacao.pdf`](docs/artigo_combustiveis_inflacao.pdf). Este repositório reúne todo o código, os dados processados, as figuras, os resultados e o modelo treinado usados no artigo.

## Resultados em resumo

| Item | Valor |
|---|---|
| Base ANP consolidada | 31 arquivos semestrais, **15.718.315 linhas** (2004 a 2026, cerca de 2,6 GB) |
| Série mensal | 180 meses com dados em 266 possíveis; as lacunas não foram interpoladas |
| Correlação de Pearson com o IPCA (169 meses) | gasolina 0,52 · diesel 0,42 · etanol 0,51 (p < 0,001) |
| Sensibilidade (peso igual por estado) | gasolina 0,52 · diesel 0,44 · etanol 0,54 |
| Melhor modelo (validação temporal) | Ridge: RMSE 0,254 p.p., MAE 0,196 p.p., R² 0,13 no teste (40 meses mais recentes) |
| Baseline (média do treino) | RMSE 0,298 p.p., R² −0,20 |

Leitura honesta: há relação real, mas modesta, entre combustíveis e IPCA. O modelo supera as baselines, porém explica pouco da variância. Detalhes e limitações no artigo.

## Mapa das etapas do CRISP-DM

| Fase | Onde está |
|---|---|
| Entendimento do negócio | artigo, seção 4.1 |
| Entendimento dos dados | `src/03_graficos_tabelas.py`, `figures/`, `results/` |
| Preparação dos dados (limpeza, filtragem, ausentes, desbalanceamento, transformação) | `src/01_consolidar_limpar.py`, `src/02_integrar_ipca.py`, `src/05_sensibilidade_regional.py`, `data/processed/` |
| Modelagem | `src/04_modelagem.py` |
| Avaliação | `src/04_modelagem.py`, `results/` |
| Implantação (opcional) | `api/main.py`, `models/modelo_final.joblib` |

## Estrutura

```
src/        scripts numerados do pipeline
api/        API FastAPI que serve o modelo final
data/processed/   dados agregados gerados pelo pipeline
figures/    gráficos usados no artigo
results/    tabelas e métricas
models/     modelo final serializado (joblib)
docs/       artigo em PDF
```

## Dados de entrada (não versionados: cerca de 2,6 GB)

- **ANP**: série histórica do levantamento de preços de combustíveis (CSVs semestrais), no portal da ANP: <https://www.gov.br/anp>
- **IBGE/SIDRA**: tabela 1737 (IPCA): <https://sidra.ibge.gov.br/tabela/1737>. O CSV já tratado está em `data/processed/ipca_mensal_1979-2026.csv`.
- Base equivalente no Base dos Dados: <https://basedosdados.org/dataset/6ea3e28a-42be-401a-a066-ad87ca931e69>

## Como reproduzir

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh /caminho/para/os/arquivos/da/anp
```

Para refazer só gráficos, modelos e a análise de sensibilidade, sem os dados brutos (os agregados já estão em `data/processed/`):

```bash
python src/02_integrar_ipca.py
python src/03_graficos_tabelas.py
python src/04_modelagem.py
python src/05_sensibilidade_regional.py
```

### API

```bash
uvicorn api.main:app --reload
curl -X POST http://127.0.0.1:8000/prever -H "Content-Type: application/json" -d '{"var_gasolina":3.0,"var_diesel":2.0,"var_etanol":1.0,"mes_calendario":3}'
```

A documentação interativa fica em `http://127.0.0.1:8000/docs`.

## Decisões metodológicas

- **Granularidade:** a ANP é semanal por posto e o IPCA é mensal nacional. Os preços foram agregados à média nacional mensal.
- **Dados ausentes:** a variação mensal só é calculada entre meses consecutivos, sem interpolação.
- **Produtos:** a modelagem usa GASOLINA, DIESEL e ETANOL.
- **Validação temporal:** treino nos meses mais antigos e teste nos 25% mais recentes, com `TimeSeriesSplit` no ajuste dos hiperparâmetros.
- **Reprodutibilidade:** semente aleatória 42.
