import importlib.metadata
import platform
import mlflow
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from xai_commun import VERSION, m, Xte, yte, CAS, context, save_json, CLASS_INDEX

run = mlflow.get_run(VERSION.run_id)
pred, proba = m.predict(Xte), m.predict_proba(Xte)[:, CLASS_INDEX]
scores = {name: float(fn(yte, pred)) for name, fn in
          [('f1', f1_score), ('precision', precision_score), ('recall', recall_score)]}
scores.update(auc=float(roc_auc_score(yte, proba)), confusion=confusion_matrix(yte, pred).tolist(),
              f1_run=float(run.data.metrics['f1']), benali_probability=float(m.predict_proba(CAS)[0, CLASS_INDEX]),
              famille=type(m).__name__, n_estimators=m.n_estimators, max_depth=m.max_depth,
              test_n=len(yte), test_fraudes=int(yte.sum()))
if abs(scores['f1'] - scores['f1_run']) > 1e-10:
    raise AssertionError('Le F1 recalculé diffère de celui du run TP2')
save_json('contexte.json', context())
save_json('verification.json', scores)
save_json('versions.json', {'python': platform.python_version(), **{p: importlib.metadata.version(p)
          for p in ['mlflow','scikit-learn','numpy','pandas','scipy','skops','shap','lime','matplotlib']}})
print('Modèle : fraude v' + VERSION.version + ' | alias production | run ' + VERSION.run_id)
for key, value in scores.items():
    print(f'{key:20s}: {value}')
print('F1 conforme au run du TP2.')
