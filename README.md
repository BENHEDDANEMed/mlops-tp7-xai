# TP07 - Explicabilité et auditabilité

Projet réalisé pour Mohamed BENHEDDANE, 5IASD, G8 (identité reprise du rapport TP2).

## Résultats vérifiés

- Véritable modèle `fraude` v1, alias `production`, forêt 500 arbres / profondeur 16.
- Run TP2 : `c982aede0f434790b4c4bc90bd0740c7`.
- F1 : 0,866920 ; précision : 0,883721 ; rappel : 0,850746 ; AUC : 0,987191.
- Matrice : `[[1101, 15], [20, 114]]` ; test de 1 250 transactions dont 134 fraudes.
- Mme Benali : probabilité 0,924 ; SHAP : nuit +0,369682, montant +0,366533,
  ancienneté +0,090302, fréquence -0,018253 ; efficacité vérifiée.
- Logistique avec log10(montant) : F1 0,859316 ; 113 fraudes détectées.
- LIME (2 000 voisins, graine 0) : montant +0,481643, nuit +0,183234,
  ancienneté +0,025545, fréquence +0,002080 ; R² local 0,494655.
- Run XAI : `1ccd984e41b94f87a354b287d5eb1269`, expérience `tp7-explicabilite`.

## Dossier et adaptation du registre

Le projet est dans `C:\Users\pc\Documents\mlops-workspace\tp7-xai`, comme prévu dans le sujet.
La base TP2 a été copiée dans `tp7-xai/mlflow.db` : le registre original reste inchangé.
Les modèles, leurs versions, les alias et les identifiants de run ont été conservés.
Les explications et la Model Card ont été ajoutées **dans cette copie TP7**.
Il ne s'agit pas d'une reconstruction du modèle ni d'un réentraînement.

La base avait déjà été ouverte avec MLflow 3.18.0 au TP6. Le serveur utilise donc
`tp6-workflows/.venv` (MLflow 3.18.0). Le client TP7 conserve MLflow 3.16.1 et toutes
les versions de sérialisation exigées dans `requirements_modele.txt`, obtenu avec
`mlflow.pyfunc.get_model_dependencies('models:/fraude@production')`.
SHAP 0.54.1 et LIME 0.2.0.1 fonctionnent avec le Python 3.14.7 installé.

Les artefacts du modèle sont toujours dans `tp2-mlflow/mlartifacts` : conserver ce
dossier et ses chemins. La copie de la base ne suffit pas pour transférer le projet
sur un autre poste. Les nouvelles explications sont dans `tp7-xai/mlartifacts`.
Une sauvegarde de la base avant les ajouts est dans `reports/mlflow-avant-tp7.db`.

## Reproduire (PowerShell)

Terminal 1 :

```powershell
cd C:\Users\pc\Documents\mlops-workspace\tp7-xai
.\demarrer_mlflow.ps1
```

Si le serveur est déjà accessible sur le port 5000, ne pas en démarrer un second.

Terminal 2 :

```powershell
cd C:\Users\pc\Documents\mlops-workspace\tp7-xai
.\executer_tp7.ps1
```

Les scripts utilisent le même test que le TP2 (`test_size=0.25`, `random_state=42`,
sans stratification). Ils résolvent l'alias puis chargent la version exacte pour
éviter un changement de modèle entre la résolution et le chargement. Le contexte
enregistré dans `reports/contexte.json` est vérifié : si les données ou le modèle
changent, conserver les résultats existants dans un autre dossier avant de refaire
les activités, plutôt que mélanger des explications de versions différentes.

Pour réarchiver les résultats et mettre à jour la carte :

```powershell
.\.venv\Scripts\python.exe src/journaliser.py
.\.venv\Scripts\python.exe preparer_documents.py
.\.venv\Scripts\python.exe src/publier_card.py
```

Cette séquence crée un **nouveau** run XAI et met à jour le lien de la version vers
ce run. Les runs précédents restent conservés. Le dernier identifiant se trouve
dans `reports/run_xai.txt` ; utiliser ce fichier plutôt qu'un identifiant copié.

## Fichiers

- `src/` : neuf scripts des activités obligatoires 1 à 7.
- `reports/` : figures SHAP et permutation, page HTML LIME, résultats JSON/CSV,
  sorties enregistrées, versions et contexte d'audit.
- `captures/` : captures réelles du registre MLflow et de l'explication LIME.
- `MODEL_CARD.md` : carte attachée à la version et au run XAI.
- `rapport_reponses.md` : réponses justifiées aux questions 1 à 10 et tableaux.
- `BENHEDDANE_Mohamed_5IASD_G8.pdf` : rapport final avec les six groupes de captures.
- `requirements-lock.txt` : environnement Python complet de cette exécution.

Les fichiers `data/`, `.venv/`, les bases et `mlartifacts/` sont exclus de Git.
Le sujet fourni par l'enseignant n'est pas republié dans le dépôt.
Les activités bonus C1/C2 ne sont pas réalisées.

L'aperçu HTML LIME est resté vide dans le navigateur disponible. Le rapport utilise
une capture du graphique PNG archivé et le précise. Le fichier HTML interactif est
livré : l'ouvrir localement et ajouter sa capture pour respecter exactement la
Capture 4 du sujet. Les sorties de terminal sont reproduites depuis les journaux
de calcul, sans fabrication de captures de terminal.

## Rendu

Le dépôt est **local**, avec les activités en commits distincts. Aucun dépôt public
GitHub n'a été créé, aucun fichier n'a été envoyé par e-mail. Après avoir choisi
le compte et confirmé la publication, le dépôt attendu se nomme `mlops-tp7-xai`.
Le rapport final doit alors être accompagné de son URL publique.

## Références

- Sujet TP07 : activités 1 à 7, questions 1 à 10.
- [Notes de version officielles SHAP](https://shap.readthedocs.io/en/stable/release_notes.html).

Les explications ne sont pas des preuves causales. Les résultats concernent les
données synthétiques du TP2 ; la carte précise les limites d'un usage réel.

Le rapport PDF et `rapport_reponses.md` restent uniquement sur le PC : ils sont exclus du dépôt et de son historique destiné à GitHub.
