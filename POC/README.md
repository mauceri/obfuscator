# POC — Historique gelé (issu de Secretarius)

Ce répertoire regroupe les trois répertoires de preuve de concept
initialement développés dans le dépôt `Secretarius`
(`aloepri_poc/`, `aloepri_freq_attack/`, `aloepri_modal/`), rapatriés ici
tels quels le 2026-09-05 avant leur suppression de Secretarius.

**Statut : figé, non maintenu.** Le travail actif sur AloePri se poursuit
dans `aloepri/`, `artifacts/`, `tools/` et `notebooks/` à la racine de ce
dépôt, qui ont depuis divergé de ce POC (corrections mathématiques,
nouvelles attaques `vma_attack`/`vma_product`/`chained_transform`,
procédure de reprise après sinistre). Ce répertoire n'est pas mis à jour
en parallèle du code actif — il sert de référence historique pour :

- `aloepri_poc/` : POC initial (défense par permutation, attaques ISA,
  mesures de qualité/vitesse) — voir `RESULTATS*.md`, `CONCLUSION.md`,
  `RUNBOOK.md`.
- `aloepri_freq_attack/` : expérience d'attaque par fréquence, non reprise
  ailleurs dans ce dépôt.
- `aloepri_modal/` : premier déploiement Modal (app, client, notebooks),
  antérieur à `modal_app.py` actuel.
