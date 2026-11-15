from pathlib import Path
RAIZ = Path(__file__).resolve().parents[1]
PROCESSADOS = RAIZ / 'data' / 'processed'
FIGURAS = RAIZ / 'figures'
RESULTADOS = RAIZ / 'results'
MODELOS = RAIZ / 'models'
PRODUTOS_ALVO = ['GASOLINA', 'DIESEL', 'ETANOL']
SEMENTE = 42
