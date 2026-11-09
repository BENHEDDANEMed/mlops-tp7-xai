import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from xai_commun import m, X, Xtr, Xte, ytr, yte, CAS, REPORTS, save_json

res = {}
def note(nom, model, test):
    pred = model.predict(test)
    res[nom] = {'f1': float(f1_score(yte, pred)), 'precision': float(precision_score(yte, pred)),
                'recall': float(recall_score(yte, pred)), 'confusion': confusion_matrix(yte, pred).tolist(),
                'fraudes_detectees': int(((pred == 1) & (yte == 1)).sum())}
    print(f"{nom:39s} F1 = {res[nom]['f1']:.4f} | fraudes : {res[nom]['fraudes_detectees']} / {int(yte.sum())}")

def logm(frame):
    frame = frame.copy()
    frame['log_montant'] = np.log10(frame.pop('montant'))
    return frame

note('foret fraude@production', m, Xte)
raw = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
note('logistique, montant brut', raw, Xte)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(logm(Xtr), ytr)
note('logistique, log10(montant)', lr, logm(Xte))
coef = dict(zip(logm(Xtr).columns, lr[-1].coef_[0].tolist()))
odds = dict(zip(logm(Xtr).columns, np.exp(lr[-1].coef_[0] / lr[0].scale_).tolist()))
print('Coefficients standardisés :', coef)
print('Multiplicateurs de cote par unité :', odds)
for depth in [2,3,6,None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr,ytr)
    note(f'arbre profondeur {depth} ({tree.get_n_leaves()} feuilles)', tree, Xte)
t3 = DecisionTreeClassifier(max_depth=3, random_state=0).fit(Xtr,ytr)
text = export_text(t3, feature_names=list(X.columns), decimals=1)
print(text)
path = []
for node in t3.decision_path(CAS).indices:
    feature = t3.tree_.feature[node]
    if feature >= 0:
        name = X.columns[feature]
        threshold = float(t3.tree_.threshold[node])
        path.append({'feature': name, 'value': float(CAS.iloc[0][name]), 'threshold': threshold,
                     'operator': '<=' if CAS.iloc[0][name] <= threshold else '>'})
print('Chemin Benali :', path, '| feuille', int(t3.apply(CAS)[0]))
(REPORTS/'arbre_profondeur3.txt').write_text(text, encoding='utf-8')
save_json('interpretables.json',res)
save_json('regles_interpretables.json', {'coefficients': coef,'odds_multipliers': odds,
          'benali_tree_path': path,'benali_leaf': int(t3.apply(CAS)[0]),
          'benali_tree_probability': float(t3.predict_proba(CAS)[0,1])})
