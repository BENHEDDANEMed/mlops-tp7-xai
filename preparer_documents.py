"""Rédiger les réponses et la Model Card à partir des résultats exécutés."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT/'reports'
def read(name):
    return json.loads((R/name).read_text(encoding='utf-8'))
v, c, a = read('verification.json'), read('contexte.json'), read('archivage.json')
local, lime = read('shap_local.json'), read('lime_reference.json')
models, rules = read('interpretables.json'), read('regles_interpretables.json')
stability = pd.read_csv(R/'lime_stabilite.csv')
imp = pd.read_json(R/'importance.json')
global_imp = pd.Series(read('shap_importance.json')).sort_values(ascending=False)
data = pd.read_csv(ROOT/'data/transactions.csv')
benali = local['benali']
comparison = []
for col in ['montant','nuit','anciennete','freq_24h']:
    row = {'variable':col,'SHAP':benali['contributions'][col],'LIME':lime['weights'][col]}
    for n in [500,2000]:
        values = stability.loc[stability.num_samples == n,col]
        row[f'min_{n}'] = float(values.min())
        row[f'max_{n}'] = float(values.max())
        row[f'signe_stable_{n}'] = bool((values > 0).all() or (values < 0).all())
    comparison.append(row)
(R/'comparaison_shap_lime.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
phrase = ('Votre paiement de 1 800 MAD a été retenu pour vérification principalement parce qu’il a été effectué '
          'de nuit pour un montant élevé ; ces éléments ont accru le risque estimé par notre système, sans prouver une fraude.')
(R/'phrase_benali.txt').write_text(phrase,encoding='utf-8')
card = f'''# Model Card - fraude v{c['modele_version']} (alias production)

## Modèle
Registre MLflow : fraude, version {c['modele_version']}, alias production.
RandomForestClassifier : {v['n_estimators']} arbres, profondeur {v['max_depth']}, random_state=42, seuil de classe 0,5.
Run d'origine : {c['modele_run_id']}.

## Objectif et usage prévu
Exercice pédagogique de détection de transactions suspectes, utilisant montant (MAD), ancienneté (mois), fréquence sur 24 h et indicateur nuit. Toute décision réelle exigerait une validation adaptée et une revue humaine.

## Usages exclus
Blocage définitif de comptes ou accusation de fraude sans revue humaine ; utilisation pour d'autres décisions financières. La précision de {v['precision']:.2%} sur le test implique {v['confusion'][0][1]} alertes injustifiées.

## Données
data/transactions.csv : {len(data)} lignes, {int(data.fraude.sum())} fraudes ({data.fraude.mean():.2%}), données synthétiques du TP2 (graine 7).
MD5 : {c['donnees_md5']}.
Découpage du TP2 : test_size=0.25, random_state=42, sans stratification ; {v['test_n']} lignes de test dont {v['test_fraudes']} fraudes. Empreinte SHA-256 et indices SHAP archivés avec le run.

## Performances
Sur le jeu de test : F1 {v['f1']:.4f}, précision {v['precision']:.4f}, rappel {v['recall']:.4f}, AUC {v['auc']:.4f}.
Matrice [[VN, FP], [FN, VP]] : {v['confusion']}. {v['confusion'][1][0]} fraudes manquées ; scores identiques au run TP2.

## Méthodes d'explicabilité
Globale : permutation F1 (10 répétitions, graine 0) et SHAP TreeExplainer (400 transactions, graine 0, classe fraude).
Locale : SHAP pour Mme Benali et test_668 ; LIME sur Mme Benali (500/2000 voisins, graines 0 à 4). Importance SHAP globale : {', '.join(global_imp.index)}.
Benali : probabilité {v['benali_probability']:.1%}, contributions principales nuit {benali['contributions']['nuit']:+.4f}, montant {benali['contributions']['montant']:+.4f} ; les contributions SHAP sont en probabilité.

## Comparaison avec un modèle interprétable
Logistique avec log10(montant) : F1 {models['logistique, log10(montant)']['f1']:.4f}, {models['logistique, log10(montant)']['fraudes_detectees']} fraudes détectées, contre {v['f1']:.4f} et 114 pour la forêt. La logistique rend les coefficients et effets sur les cotes accessibles ; ce test seul ne démontre pas une différence significative.

## Limites et risques connus
Données synthétiques et petit effectif de fraudes ; aucune validation sur transactions réelles, sous-groupes, dérive ou calibration. SHAP explique les prédictions, sans causalité ; LIME dépend du voisinage et des graines. Le score du substitut local LIME (2000 voisins, graine 0) est R²={lime['r2_local']:.3f}.

## Recommandations d'utilisation et de surveillance
Revue humaine des alertes ; surveiller rappel, précision, faux positifs, calibration et dérive des quatre variables. Recalculer les explications et revoir cette carte après réentraînement ou changement d'alias, avant validation de la nouvelle version.

## Traçabilité
Expérience : tp7-explicabilite ; run XAI : {a['run_id']}.
Registre de travail : copie du registre TP2 dans tp7-xai/mlflow.db, servie avec MLflow 3.18.0 (environnement TP6) ; client et dépendances du modèle : MLflow 3.16.1, scikit-learn 1.9.1. Même version, modèle et run d'origine ; aucun réentraînement ni changement de l'alias.
Le registre original du TP2 reste inchangé ; les nouveaux tags et la description sont dans la copie TP7. Les artefacts du modèle restent référencés dans le TP2 : conserver les deux dossiers.
Auteurs : Mohamed BENHEDDANE, 5IASD, G8 (identité reprise du rapport TP2). Date : 09/11/2026. Prochaine revue proposée : 09/12/2026, ou avant tout nouvel usage.
'''
(ROOT/'MODEL_CARD.md').write_text(card,encoding='utf-8')
questions = [
('1. Le registre et l’alias',f"models:/fraude@production retrouve la version validée sans figer son numéro dans le service. Ici, production désigne v{c['modele_version']} (run {c['modele_run_id']}). Si l’alias passe à v2, les explications archivées restent celles de v1 : les tags modele_version, modele_run_id et l’empreinte des données le montrent. Il faut refaire les explications pour v2. Les scripts résolvent l’alias puis chargent la version exacte, et refusent un contexte différent pour éviter de mélanger des résultats."),
('2. Deux importances globales',f"Le classement MDI est montant, anciennete, freq_24h, nuit ; la permutation donne montant, freq_24h, anciennete, nuit. Les deux variables centrales de la permutation sont presque à égalité ({imp.loc['freq_24h','permutation']:.4f} et {imp.loc['anciennete','permutation']:.4f}) par rapport aux écarts-types. La MDI cumule les diminutions d’impureté des séparations apprises, normalisées, et favorise souvent les variables offrant beaucoup de seuils possibles. La permutation mesure sur le test la perte de F1 après mélange : nuit est binaire mais sa permutation fait baisser le F1 de {imp.loc['nuit','permutation']:.4f}, alors que sa MDI vaut {imp.loc['nuit','MDI (foret)']:.4f}. Ces nombres n’ont ni la même échelle ni la même définition ; leur différence ne mesure pas un biais à elle seule."),
('3. Lire le beeswarm',"Une faible ancienneté contribue généralement positivement au score de fraude (points bleus à droite), une forte ancienneté négativement (points rouges à gauche). Le beeswarm montre le sens, la dispersion et l’hétérogénéité des contributions pour chaque observation. La permutation ne fournit qu’une perte de score agrégée, sans dire si les petites ou grandes valeurs augmentent le risque. Le nuage du montant fait apparaître une transition autour de 1 000 MAD, variable selon le contexte ; ce n’est pas une règle unique de blocage."),
('4. Global et local',f"Il n’y a pas de contradiction : l’importance globale moyenne |SHAP| sur 400 transactions, alors que l’explication locale porte sur un cas. Pour Benali, nuit est première ({benali['contributions']['nuit']:+.6f}), suivie du montant ({benali['contributions']['montant']:+.6f}), alors que nuit arrive troisième en moyenne. Efficacité : {benali['base_value']:.6f} + ({benali['contributions']['nuit']:.6f} + {benali['contributions']['montant']:.6f} + {benali['contributions']['anciennete']:.6f} - {abs(benali['contributions']['freq_24h']):.6f}) = {benali['base_plus_sum']:.6f}, soit la prédiction {benali['probability']:.6f}. Les écarts résiduels sont de l’ordre de l’arrondi numérique."),
('5. La phrase pour la cliente',phrase+" Pour test_668, nuit=0 : l’absence de paiement nocturne fait diminuer le score par rapport à la référence de la forêt (SHAP nuit=-0,040959). Le montant, la faible ancienneté et la fréquence suffisent toutefois à produire une probabilité de 0,996. Il s’agit d’un effet prédictif relatif à la référence, sans interprétation causale."),
('6. La stabilité de LIME','Les intervalles par variable et par taille de voisinage sont donnés dans le tableau de stabilité. '+ '; '.join(f"{row['variable']} : signe {'stable' if row['signe_stable_500'] else 'variable'} à 500 voisins, {'stable' if row['signe_stable_2000'] else 'variable'} à 2000" for row in comparison)+". Avant de communiquer une explication, répéter les graines, augmenter le voisinage si nécessaire, vérifier la fidélité du substitut et retenir les causes robustes. Un coefficient proche de zéro ne doit pas être présenté comme une cause ferme. Cinq graines restent un contrôle limité, et stabilité ne garantit pas fidélité."),
('7. SHAP ou LIME',f"Les deux méthodes identifient montant et nuit comme facteurs majeurs pour Benali, avec un ordre susceptible de différer. SHAP attribue l’écart à une valeur de base du modèle ; LIME ajuste un substitut local sur des perturbations et des règles discrétisées. Leurs coefficients n’ont donc pas la même signification et ne se comparent pas directement. Le substitut LIME a ici R²={lime['r2_local']:.3f}, et sa prédiction locale vaut {lime['local_prediction']:.3f}, contre 0,924 pour la forêt. Pour cette forêt, je retiens TreeExplainer : méthode adaptée aux arbres, contributions reproductibles pour une configuration fixe, avec efficacité vérifiée."),
('8. Le compromis',f"Pour une validation en vue du blocage de paiements, je retiendrais la logistique avec log10(montant) comme candidate prioritaire grâce à sa lisibilité : elle détecte {models['logistique, log10(montant)']['fraudes_detectees']}/134 fraudes contre 114/134 pour la forêt, avec F1={models['logistique, log10(montant)']['f1']:.4f} contre {v['f1']:.4f}, soit un écart de {v['f1']-models['logistique, log10(montant)']['f1']:.4f}. La matrice de confusion permet aussi de comparer les fausses alertes. Un seul test de 134 fraudes ne permet pas de conclure à une supériorité robuste : il faut une comparaison appariée, des validations temporelles, les coûts des erreurs et la calibration. Les données étant synthétiques, aucune mise en production réelle n’est justifiée par ce TP seul. Un arbre de profondeur 3 est lisible mais perd nettement en détection."),
('9. Reconstituer une décision',f"Le run XAI {a['run_id']} conserve version/run du modèle, empreintes MD5 et SHA-256, découpage, indices SHAP, caractéristiques de Benali, contributions, graines, réglages LIME, bibliothèques, scripts et graphiques. Le tag explications_run relie la version à ce run. L’empreinte permet de vérifier un fichier mais ne le restaure pas : il faut conserver les données et les artefacts du TP2, ou une version DVC récupérable. Ce TP n’enregistre pas le journal immuable de la décision réellement servie (identifiant de transaction, horodatage, vecteur d’entrée et transformations, seuil/politique et version effectivement utilisée). Ce journal manque pour un audit opérationnel complet."),
('10. La Model Card',f"Usage exclu : le blocage définitif d’un compte sans revue humaine, car {v['confusion'][0][1]} paiements normaux sont signalés à tort sur le test (précision {v['precision']:.4f}). Limite : les {len(data)} transactions sont synthétiques, et {v['confusion'][1][0]} fraudes restent manquées ; ces résultats ne démontrent ni la performance sur de vraies transactions ni l’équité par sous-groupe. La carte doit donc documenter cette origine et imposer une validation adaptée avant un usage réel.")]
(R/'reponses.json').write_text(json.dumps(questions,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# TP07 - Explicabilité et auditabilité','', 'Mohamed BENHEDDANE - 5IASD - G8 - 09/11/2026','',
       'Activités obligatoires 1 à 7 réalisées. C1/C2 (bonus) non réalisés. Dépôt Git local ; publication GitHub et envoi du rapport non effectués.','']
for title, answer in questions:
    lines.extend(['## Question '+title,'',answer,''])
lines.extend(['## Tableau SHAP / LIME','', '| Variable | SHAP | LIME (2000, graine 0) | Signe stable (2000) |','|---|---:|---:|---|'])
for row in comparison:
    lines.append(f"| {row['variable']} | {row['SHAP']:+.6f} | {row['LIME']:+.6f} | {'Oui' if row['signe_stable_2000'] else 'Non'} |")
lines.extend(['','## Compromis','', '| Modèle | F1 | Fraudes détectées / 134 |','|---|---:|---:|'])
for name, score in models.items():
    lines.append(f"| {name} | {score['f1']:.4f} | {score['fraudes_detectees']} |")
lines.extend(['','## Chemin de Mme Benali dans l’arbre de profondeur 3',''])
lines.append(' ET '.join(f"{p['feature']} {p['operator']} {p['threshold']:.2f}" for p in rules['benali_tree_path'])+f" => feuille {rules['benali_leaf']}, P(fraude)={rules['benali_tree_probability']:.3f}.")
lines.extend(['','## Arbre complet','', '```text', (R/'arbre_profondeur3.txt').read_text(encoding='utf-8'),'```','','## Model Card','',card])
(ROOT/'rapport_reponses.md').write_text('\n'.join(lines),encoding='utf-8')
print('MODEL_CARD.md, rapport_reponses.md et tableaux préparés à partir des résultats.')
