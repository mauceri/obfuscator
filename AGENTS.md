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
  chirurgicales, exécution orientée objectif, ne jamais entreprendre de tâche
  longue sans accord préalable).
- Communication en français ; les termes techniques (paper, APIs) restent en anglais.
- Avant d'implémenter une brique d'AloePri, énoncer explicitement les hypothèses
  de lecture du papier et les confirmer.
- Chaque étape se termine par une vérification reproductible (test ou script de mesure).

## Références

- Papier : https://arxiv.org/pdf/2603.01499
