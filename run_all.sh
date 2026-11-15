#!/usr/bin/env bash
set -e
ENTRADA="${1:?Informe a pasta com os .zip/.csv semestrais da ANP}"
python src/01_consolidar_limpar.py --entrada "$ENTRADA"
python src/02_integrar_ipca.py
python src/03_graficos_tabelas.py
python src/04_modelagem.py
python src/05_sensibilidade_regional.py
