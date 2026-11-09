# Model Card - fraude v1 (alias production)

## Modèle
Registre MLflow : fraude, version 1, alias production.
RandomForestClassifier : 500 arbres, profondeur 16, random_state=42, seuil de classe 0,5.
Run d'origine : c982aede0f434790b4c4bc90bd0740c7.

## Objectif et usage prévu
Exercice pédagogique de détection de transactions suspectes, utilisant montant (MAD), ancienneté (mois), fréquence sur 24 h et indicateur nuit. Toute décision réelle exigerait une validation adaptée et une revue humaine.

## Usages exclus
Blocage définitif de comptes ou accusation de fraude sans revue humaine ; utilisation pour d'autres décisions financières. La précision de 88.37% sur le test implique 15 alertes injustifiées.

## Données
data/transactions.csv : 5000 lignes, 568 fraudes (11.36%), données synthétiques du TP2 (graine 7).
MD5 : cf9f6f86c141e7df32c61715c493483f.
Découpage du TP2 : test_size=0.25, random_state=42, sans stratification ; 1250 lignes de test dont 134 fraudes. Empreinte SHA-256 et indices SHAP archivés avec le run.

## Performances
Sur le jeu de test : F1 0.8669, précision 0.8837, rappel 0.8507, AUC 0.9872.
Matrice [[VN, FP], [FN, VP]] : [[1101, 15], [20, 114]]. 20 fraudes manquées ; scores identiques au run TP2.

## Méthodes d'explicabilité
Globale : permutation F1 (10 répétitions, graine 0) et SHAP TreeExplainer (400 transactions, graine 0, classe fraude).
Locale : SHAP pour Mme Benali et test_668 ; LIME sur Mme Benali (500/2000 voisins, graines 0 à 4). Importance SHAP globale : montant, anciennete, nuit, freq_24h.
Benali : probabilité 92.4%, contributions principales nuit +0.3697, montant +0.3665 ; les contributions SHAP sont en probabilité.

## Comparaison avec un modèle interprétable
Logistique avec log10(montant) : F1 0.8593, 113 fraudes détectées, contre 0.8669 et 114 pour la forêt. La logistique rend les coefficients et effets sur les cotes accessibles ; ce test seul ne démontre pas une différence significative.

## Limites et risques connus
Données synthétiques et petit effectif de fraudes ; aucune validation sur transactions réelles, sous-groupes, dérive ou calibration. SHAP explique les prédictions, sans causalité ; LIME dépend du voisinage et des graines. Le score du substitut local LIME (2000 voisins, graine 0) est R²=0.495.

## Recommandations d'utilisation et de surveillance
Revue humaine des alertes ; surveiller rappel, précision, faux positifs, calibration et dérive des quatre variables. Recalculer les explications et revoir cette carte après réentraînement ou changement d'alias, avant validation de la nouvelle version.

## Traçabilité
Expérience : tp7-explicabilite ; run XAI : 1ccd984e41b94f87a354b287d5eb1269.
Registre de travail : copie du registre TP2 dans tp7-xai/mlflow.db, servie avec MLflow 3.18.0 (environnement TP6) ; client et dépendances du modèle : MLflow 3.16.1, scikit-learn 1.9.1. Même version, modèle et run d'origine ; aucun réentraînement ni changement de l'alias.
Le registre original du TP2 reste inchangé ; les nouveaux tags et la description sont dans la copie TP7. Les artefacts du modèle restent référencés dans le TP2 : conserver les deux dossiers.
Auteurs : Mohamed BENHEDDANE, 5IASD, G8 (identité reprise du rapport TP2). Date : 09/11/2026. Prochaine revue proposée : 09/12/2026, ou avant tout nouvel usage.
