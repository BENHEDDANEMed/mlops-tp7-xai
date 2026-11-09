"""Chargement du vrai registre TP2 et contexte reproductible du TP7."""
import hashlib
import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
NOM, ALIAS = 'fraude', 'production'
VERSION = MlflowClient().get_model_version_by_alias(NOM, ALIAS)
# Résoudre l'alias, puis charger cette version exacte évite une course si l'alias change.
m = mlflow.sklearn.load_model(f'models:/{NOM}/{VERSION.version}')
df = pd.read_csv(ROOT / 'data/transactions.csv')
X, y = df.drop(columns='fraude'), df['fraude']
if list(X.columns) != list(m.feature_names_in_):
    raise ValueError('Ordre des variables différent de celui du modèle TP2')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
CAS = pd.DataFrame([{'montant': 1800.0, 'anciennete': 30, 'freq_24h': 4, 'nuit': 1}])[X.columns]
CLASS_INDEX = list(m.classes_).index(1)


def save_json(name, value):
    (REPORTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def context():
    return {'modele': NOM, 'alias': ALIAS, 'modele_version': VERSION.version,
            'modele_run_id': VERSION.run_id, 'donnees': 'data/transactions.csv',
            'donnees_md5': hashlib.md5((ROOT / 'data/transactions.csv').read_bytes()).hexdigest(),
            'donnees_sha256': hashlib.sha256((ROOT / 'data/transactions.csv').read_bytes()).hexdigest(),
            'test_size': 0.25, 'split_random_state': 42}


def check_context():
    path = REPORTS / 'contexte.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != context():
        raise RuntimeError('Le modèle ou les données ont changé depuis la vérification. Refaire les activités dans un nouveau dossier de résultats.')


def fraud_shap(explainer, frame):
    values = explainer(frame)
    if values.values.ndim != 3:
        raise ValueError('Sortie SHAP inattendue pour cette forêt binaire : vérifier la classe expliquée')
    return values[:, :, CLASS_INDEX]


def predict_proba(values):
    return m.predict_proba(pd.DataFrame(values, columns=X.columns))


check_context()
