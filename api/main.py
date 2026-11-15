from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
CAMINHO = Path(__file__).resolve().parents[1] / 'models' / 'modelo_final.joblib'
pacote = joblib.load(CAMINHO)
app = FastAPI(title='Combustiveis vs Inflacao', version='1.0', description='Estima o IPCA mensal (%) a partir da variacao mensal (%) dos precos dos combustiveis.')

class Entrada(BaseModel):
    var_gasolina: float = Field(..., description='Variacao % mensal do preco medio da gasolina')
    var_diesel: float = Field(..., description='Variacao % mensal do preco medio do diesel')
    var_etanol: float = Field(..., description='Variacao % mensal do preco medio do etanol')
    var_gasolina_lag1: float = Field(0.0, description='Variacao % da gasolina no mes anterior')
    var_diesel_lag1: float = Field(0.0, description='Variacao % do diesel no mes anterior')
    var_etanol_lag1: float = Field(0.0, description='Variacao % do etanol no mes anterior')
    ipca_lag1: float = Field(0.4, description='IPCA (%) do mes anterior')
    mes_calendario: int = Field(..., ge=1, le=12, description='Mes do ano (1-12)')

@app.get('/')
def raiz():
    return {'modelo': pacote['nome'], 'atributos': pacote['atributos'], 'docs': '/docs'}

@app.post('/prever')
def prever(x: Entrada):
    linha = pd.DataFrame([x.model_dump()])[pacote['atributos']]
    return {'ipca_mensal_previsto_pct': round(float(pacote['modelo'].predict(linha)[0]), 3), 'modelo': pacote['nome']}
