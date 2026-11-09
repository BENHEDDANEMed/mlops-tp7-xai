import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shap
from xai_commun import m, Xte, yte, CAS, REPORTS, fraud_shap, save_json, CLASS_INDEX

ex = shap.TreeExplainer(m)
idx = Xte.index[m.predict_proba(Xte)[:, CLASS_INDEX].argmax()]
results = {}
for nom, x in [('benali', CAS), (f'test_{idx}', Xte.loc[[idx]])]:
    e = fraud_shap(ex, x)[0]
    probability = float(m.predict_proba(x)[0, CLASS_INDEX])
    total = float(e.base_values + e.values.sum())
    assert np.isclose(total, probability, atol=1e-6)
    results[nom] = {'features': x.iloc[0].to_dict(), 'base_value': float(e.base_values),
                    'contributions': dict(zip(x.columns, e.values.tolist())),
                    'base_plus_sum': total, 'probability': probability,
                    'true_label': None if nom == 'benali' else int(yte.loc[idx])}
    print('\n---', nom, results[nom]['features'])
    print('Valeur de base :', round(float(e.base_values), 6))
    for col, value in sorted(results[nom]['contributions'].items(), key=lambda p: -abs(p[1])):
        print(f'{col:12s} {value:+.6f}')
    print('Base + somme :', total, '| prédiction :', probability)
    shap.plots.waterfall(e, show=False)
    plt.tight_layout()
    plt.savefig(REPORTS / f'shap_{nom}.png', dpi=160)
    plt.close('all')
save_json('shap_local.json', results)
