import json
import mlflow
from mlflow import MlflowClient
from xai_commun import ROOT, REPORTS, NOM, VERSION

receipt = json.loads((REPORTS/'archivage.json').read_text(encoding='utf-8'))
run_id = (REPORTS/'run_xai.txt').read_text(encoding='utf-8').strip()
if receipt['model_version'] != VERSION.version or receipt['model_run_id'] != VERSION.run_id:
    raise RuntimeError('La version expliquée ne correspond plus à l’alias production')
card = (ROOT/'MODEL_CARD.md').read_text(encoding='utf-8')
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(str(ROOT/'MODEL_CARD.md'))
client = MlflowClient()
client.update_model_version(NOM, receipt['model_version'], description=card)
client.set_model_version_tag(NOM, receipt['model_version'],'model_card','MODEL_CARD.md, run '+run_id)
assert client.get_model_version(NOM, receipt['model_version']).description == card
print('Model Card attachée à fraude v'+receipt['model_version']+' et au run '+run_id)
