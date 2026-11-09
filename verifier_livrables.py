"""Contrôles de cohérence des livrables et de la traçabilité MLflow."""
import hashlib
import json
from pathlib import Path
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT/'reports'
mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = MlflowClient()
def read(name): return json.loads((R/name).read_text(encoding='utf-8'))
context, metrics, receipt = read('contexte.json'),read('verification.json'),read('archivage.json')
version = client.get_model_version_by_alias('fraude','production')
assert version.version == receipt['model_version'] == context['modele_version']
assert version.run_id == receipt['model_run_id'] == context['modele_run_id']
run = client.get_run(receipt['run_id'])
assert run.info.status == 'FINISHED'
assert version.tags['explications_run'] == receipt['run_id']
assert version.description == (ROOT/'MODEL_CARD.md').read_text(encoding='utf-8')
assert 'MODEL_CARD.md' in version.tags['model_card']
assert abs(metrics['f1'] - metrics['f1_run']) < 1e-12
assert run.data.tags['donnees_md5'] == hashlib.md5((ROOT/'data/transactions.csv').read_bytes()).hexdigest()
original = ROOT.parent/'tp2-mlflow/mlflow.db'
backup = R/'mlflow-avant-tp7.db'
assert hashlib.sha256(original.read_bytes()).digest() == hashlib.sha256(backup.read_bytes()).digest()
assert hashlib.sha256((ROOT/'data/transactions.csv').read_bytes()).digest() == hashlib.sha256((ROOT.parent/'tp2-mlflow/data/transactions.csv').read_bytes()).digest()
for explanation in read('shap_local.json').values():
    assert np.isclose(explanation['base_value']+sum(explanation['contributions'].values()),explanation['probability'],atol=1e-9)
assert read('shap_echantillon.json')['additivity_max_error'] < 1e-9
rows = read('lime_stabilite.json')
assert {(row['num_samples'],row['seed']) for row in rows} == {(n,s) for n in [500,2000] for s in range(5)}
assert len(read('reponses.json')) == 10
for file in ['xai_commun.py','verifier_modele.py','xai_importance.py','xai_global.py','xai_local.py','xai_lime.py','interpretables.py','journaliser.py','publier_card.py']:
    assert (ROOT/'src'/file).exists(), file
for file in ['shap_global.png','shap_benali.png','shap_test_668.png','lime_benali.html','lime_stabilite.json','importance.json','interpretables.json']:
    artifact = next(f for f in client.list_artifacts(receipt['run_id'],'explications') if Path(f.path).name == file)
    assert artifact.file_size == (R/file).stat().st_size, file
result = {'statut':'VALIDE','run_xai':receipt['run_id'],'modele_version':version.version,
          'f1':metrics['f1'],'original_tp2_inchange':True,'donnees_identiques_tp2':True,
          'shap_additivite':True,'lime_10_experiences':True,'model_card_attachee':True,
          'limites_rendu':['Capture HTML LIME à ajouter ; équivalent PNG inclus',
                          'Publication GitHub et envoi e-mail non effectués',
                          'Activités bonus C1/C2 non réalisées']}
(R/'validation_finale.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
