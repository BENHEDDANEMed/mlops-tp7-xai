import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from xai_commun import m, X, Xte, yte, REPORTS

r = permutation_importance(m, Xte, yte, scoring='f1', n_repeats=10, random_state=0)
t = pd.DataFrame({'MDI (foret)': m.feature_importances_, 'permutation': r.importances_mean,
                  'ecart-type': r.importances_std}, index=X.columns).sort_values('permutation', ascending=False)
print(t.round(4).to_string())
t.to_json(REPORTS / 'importance.json', indent=2)
t.to_csv(REPORTS / 'importance.csv')
t['permutation'].plot.barh(xerr=t['ecart-type'], color='#4592EF')
plt.xlabel('Perte de F1 après permutation (10 répétitions)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(REPORTS / 'permutation.png', dpi=160)
