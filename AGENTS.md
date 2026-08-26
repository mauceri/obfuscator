# AGENTS.md — obfuscator

## Mission

Implémenter **AloePri**, la méthode d'inférence LLM préservant la vie privée par
**obfuscation covariante**, décrite dans :

> [Towards Privacy-Preserving LLM Inference via Covariant Obfuscation (Technical Report)](https://arxiv.org/pdf/2603.01499) — arXiv:2603.01499

AloePri transforme conjointement les **données** (entrées et sorties) et les
**paramètres du modèle** pour que l'inférence s'exécute sur des infrastructures
non fiables (cloud, clusters hétérogènes de xPU) sans exposer les données
privées, tout en restant compatible avec les infrastructures LMaaS existantes.

Les trois exigences cœur du papier :

1. Pertes de précision et d'efficacité minimales ;
2. Exécution sur de grands clusters de xPU hétérogènes et existants ;
3. Compatibilité avec les infrastructures LLM existantes (réutiliser leurs optimisations).

## Contexte

- Machine : `sanroque`
- Dépôt : https://github.com/mauceri/obfuscator
- Application cible (hypothèse à confirmer) : réutiliser AloePri pour obfusquer
  **Qwen3-8B** (évoqué lors d'une session précédente).

## Stack technique

- **Python + PyTorch** : transformations des poids et des embeddings via
  `nn.Module` ; écosystème `transformers` / `safetensors` pour charger les modèles.

## Règles de travail

- Les instructions globales de l'utilisateur s'appliquent en priorité :
  `~/.dsh/AGENTS.md` (réfléchir avant de coder, simplicité d'abord, modifications
  chirurgicales, exécution orientée objectif).
- **Consigne générale : ne JAMAIS entreprendre une tâche longue (fine-tuning,
  entraînements, runs Modal, calculs de plusieurs minutes ou plus) sans
  demander l'accord de l'utilisateur au préalable, avec le budget estimé.**
- Communication en français ; les termes techniques (paper, APIs) restent en anglais.
- Avant d'implémenter une brique d'AloePri, énoncer explicitement les hypothèses
  de lecture du papier et les confirmer.
- Chaque étape se termine par une vérification reproductible (test ou script de mesure).

## Consigne déploiement confidentiel (procédure par défaut)

Tout déploiement d'un modèle obfusqué avec AloePri suit cette procédure :

1. **Fine-tuning sur le corpus d'intérêt** (privé ou synthétique) AVANT
   l'obfuscation — les poids affinés W_ft diffèrent du modèle public : la
   référence de l'attaque VMA (appariement de vocabulaire) devient fausse et
   la récupération de la permutation Π échoue ;
2. **AloePri** — au minimum la permutation de vocabulaire ; le schéma complet
   (matrices clés h>0, `aloepri/chained_transform.py`) quand la qualité le
   permet (la VMA directe y est structurellement impossible) ;
3. **Rotation périodique de la permutation secrète** — contre les attaques
   basées sur la fréquence (TFMA/SDA du papier) qui exploitent la stabilité
   de la permutation.

Contexte sécurité (à garder en tête) :
- La VMA directe (embedding vs table claire) est impossible en h>0
  (dimensions d+2h ≠ d) ; la VMA produit (Table 9) n'est défendue QUE par le
  bruit α_e — à α_e=0 elle récupère ~100 % de Π (mesuré 2026-08-26).
- Les clés de permutation ne quittent jamais le client (posture existante).

## Références

- Papier : https://arxiv.org/pdf/2603.01499
