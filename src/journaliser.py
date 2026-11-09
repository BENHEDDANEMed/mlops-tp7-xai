import importlib.metadata
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score
from xai_commun import NOM, ALIAS, VERSION, m, Xte, yte, ROOT, REPORTS, context, save_json

pred = m.predict(Xte)
mlflow.set_experiment('tp7-explicabilite')
with mlflow.start_run(run_name=f'xai-{NOM}-v{VERSION.version}') as run:
    mlflow.set_tags({**context(), 'methodes': 'permutation ; SHAP TreeExplainer ; LIME'})
    mlflow.log_params({'shap_echantillon':400,'shap_random_state':0,'lime_num_samples':2000,
                      'lime_random_state':0,'lime_stabilite_seeds':'0,1,2,3,4',
                      'permutation_n_repeats':10,'permutation_random_state':0})
    mlflow.log_metrics({'f1':f1_score(yte,pred),'precision':precision_score(yte,pred),'recall':recall_score(yte,pred)})
    for file in sorted(REPORTS.iterdir()):
        if file.suffix in {'.png','.html','.json','.csv','.txt'} and file.name != 'run_xai.txt':
            mlflow.log_artifact(str(file),artifact_path='explications')
    for file in [ROOT/'requirements_modele.txt', ROOT/'requirements-lock.txt']:
        if file.exists(): mlflow.log_artifact(str(file),artifact_path='environnement')
    mlflow.log_artifacts(str(ROOT/'src'),artifact_path='code')
    (REPORTS/'run_xai.txt').write_text(run.info.run_id,encoding='utf-8')
MlflowClient().set_model_version_tag(NOM,VERSION.version,'explications_run',run.info.run_id)
save_json('archivage.json', {'run_id':run.info.run_id,'experiment_id':run.info.experiment_id,
                          'model_version':VERSION.version,'model_run_id':VERSION.run_id})
print('Run XAI :',run.info.run_id,'| modèle',NOM,'v'+VERSION.version)
