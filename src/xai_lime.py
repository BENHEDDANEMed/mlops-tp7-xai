import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from xai_commun import X, Xtr, CAS, REPORTS, predict_proba, save_json, CLASS_INDEX


def expliquer(seed, n):
    explainer = LimeTabularExplainer(Xtr.values, feature_names=list(X.columns),
        class_names=['normal','fraude'], mode='classification', random_state=seed)
    return explainer.explain_instance(CAS.values[0], predict_proba, labels=(CLASS_INDEX,),
                                      num_features=4, num_samples=n)


e = expliquer(0, 2000)
print('Règles :', [(rule, round(w, 4)) for rule, w in e.as_list(label=CLASS_INDEX)])
print('Score R² du substitut local :', e.score, '| estimation locale :', e.local_pred[0])
e.save_to_file(str(REPORTS / 'lime_benali.html'))
fig = e.as_pyplot_figure(label=CLASS_INDEX)
fig.tight_layout()
fig.savefig(REPORTS / 'lime_benali.png', dpi=160)
plt.close('all')
rows = []
for n in [500, 2000]:
    print(f'\n--- {n} voisins : montant / nuit / anciennete / freq_24h')
    for seed in range(5):
        exp = e if (seed, n) == (0, 2000) else expliquer(seed, n)
        # Les indices des variables sont plus fiables qu'une recherche dans les règles textuelles.
        weights = {str(X.columns[i]): float(w) for i, w in exp.as_map()[CLASS_INDEX]}
        rows.append({'num_samples': n, 'seed': seed, **weights, 'r2_local': float(exp.score),
                     'local_prediction': float(exp.local_pred[0])})
        print(f"graine {seed} : " + '  '.join(f'{weights[c]:+.4f}' for c in ['montant','nuit','anciennete','freq_24h']))
pd.DataFrame(rows).to_csv(REPORTS / 'lime_stabilite.csv', index=False)
save_json('lime_stabilite.json', rows)
save_json('lime_reference.json', {'rules': e.as_list(label=CLASS_INDEX),
          'weights': {str(X.columns[i]): float(w) for i, w in e.as_map()[CLASS_INDEX]},
          'r2_local': float(e.score), 'local_prediction': float(e.local_pred[0])})
