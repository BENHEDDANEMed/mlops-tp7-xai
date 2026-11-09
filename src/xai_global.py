import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xai_commun import m, Xte, REPORTS, fraud_shap, save_json, CLASS_INDEX

Xs = Xte.sample(400, random_state=0)
sv = fraud_shap(shap.TreeExplainer(m), Xs)
error = float(np.max(np.abs(sv.base_values + sv.values.sum(axis=1) - m.predict_proba(Xs)[:, CLASS_INDEX])))
assert error < 1e-6, error
imp = pd.Series(np.abs(sv.values).mean(0), index=Xs.columns).sort_values(ascending=False)
print('Valeur de base :', float(sv.base_values[0]))
print(imp.round(4).to_string())
print('Erreur maximale d’additivité :', error)
imp.to_json(REPORTS / 'shap_importance.json', indent=2)
save_json('shap_echantillon.json', {'indices': Xs.index.tolist(), 'additivity_max_error': error,
                                  'base_value': float(sv.base_values[0])})
pd.DataFrame(sv.values, columns=Xs.columns, index=Xs.index).to_csv(REPORTS / 'shap_valeurs.csv')
shap.plots.beeswarm(sv, show=False)
plt.tight_layout()
plt.savefig(REPORTS / 'shap_global.png', dpi=160)
plt.close('all')
plt.scatter(Xs['montant'], sv[:, 'montant'].values, c=Xs['nuit'], cmap='coolwarm', s=12)
plt.axhline(0, color='grey', linewidth=0.7)
plt.xscale('log')
plt.xlabel('Montant (MAD)')
plt.ylabel('Contribution SHAP du montant')
plt.colorbar(label='Nuit')
plt.tight_layout()
plt.savefig(REPORTS / 'shap_montant.png', dpi=160)
