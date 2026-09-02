# Rotation des clés — obfuscator (AloePri h>0)

La confidentialité du système repose sur la permutation Π (côté client) **et**
sur le modèle obfusqué servi (lignes d'embedding/head indexées par Π). Une
rotation doit donc renouveler les deux de façon synchronisée.

Deux niveaux, selon la menace :

| Niveau | Quoi | Protège contre | Coût |
|---|---|---|---|
| 1 — secours | nouvelle Π (aléatoire) + réordonnancement des lignes embed/head du modèle existant | réutilisation d'une clé compromise ; démo réutilisée avec une clé connue | rapide (disque + CPU) |
| 2 — complète | nouvelle seed → nouveaux P̂/Q̂, nouveaux bruits α_e/α_h, **modèle entièrement re-transformé** | aussi l'**appariement différentiel multi-snapshots** (serveur conservant l'ancien et le nouveau modèle) | re-transform complet (~1-1,5 h Modal CPU, ou ~48 Go RAM en local) |

> **Règle d'or** : ne **jamais** laisser une ancienne clé (ou une ancienne
> seed) en service après une rotation, et **retirer du volume** l'ancien
> modèle obfusqué. Sinon le serveur peut croiser les snapshots et relier les
> lignes entre générations (observation multi-rotations — la rotation n'a
> alors servi à rien).

---

## Niveau 1 — rotation de secours (Π seule)

**Quand** : clé soupçonnée d'exposition, démo à refaire après une clé
devenue publique (`seed 0` documentée), rotation périodique légère.

**Limite assumée** : les bruits α_e/α_h et les matrices P̂/Q̂ sont inchangés ;
un attaquant ayant conservé l'ancien modèle peut relier les lignes (appariement
différentiel). Ce niveau est un **garde-fou opérationnel**, pas une
ré-obfuscation.

1. Récupérer localement le modèle servi depuis le volume Modal :

   ```bash
   mkdir -p /tmp/rot && cd /tmp/rot
   ~/modal-venv/bin/modal volume get obfuscator-models qwen3-8b-ft-h128-a1-h02 ./qwen3-8b-ft-h128-a1-h02
   ```

2. Générer la nouvelle permutation (CSPRNG) et réordonner embed/head :

   ```bash
   cd <repo-obfuscator>
   .venv/bin/python tools/rotate_pi.py \
       --model-in  /tmp/rot/qwen3-8b-ft-h128-a1-h02 \
       --model-out /tmp/rot/qwen3-8b-ft-h128-a1-h02-r1 \
       --keys-in   artifacts/obfuscation_keys.json \
       --keys-out  artifacts/obfuscation_keys-r1.json
   ```

   → produit un modèle dont la ligne j porte le token clair `unperm_new[j]`,
   et la nouvelle clé (`seed: null` = aléatoire, jamais publiée).

3. Remplacer le modèle servi sur le volume **et supprimer l'ancien** :

   ```bash
   ~/modal-venv/bin/modal volume put obfuscator-models /tmp/rot/qwen3-8b-ft-h128-a1-h02-r1 qwen3-8b-ft-h128-a1-h02-r1
   ~/modal-venv/bin/modal volume rm -r obfuscator-models qwen3-8b-ft-h128-a1-h02   # retirer l'ancien (multi-snapshot)
   ```

   Puis pointer le service sur le nouveau sous-dossier (`MODEL_SUBDIR` dans
   modal_app.py) et `modal deploy modal_app.py`.

4. Basculer le client sur la nouvelle clé :

   ```bash
   mv artifacts/obfuscation_keys-r1.json artifacts/obfuscation_keys.json
   ```

5. Vérifier : round-trip client (« La capitale de la France est Paris ») et
   `/health`. **Jeter** l'ancienne clé et l'ancien modèle (ne pas les garder
   « au cas où » — c'est eux qui permettent l'appariement différentiel).

---

## Niveau 2 — rotation complète (re-transform)

**Quand** : rotation robuste, avant une utilisation réellement confidentielle,
ou après une exposition longue.

Une nouvelle clé Π **et** de nouveaux secrets de transformation doivent être
tirés ensemble : la permutation de vocabulaire, les matrices clés P̂/Q̂ (via
leur seed), les bruits E_embed/E_head (seeds dérivées), et κ. Tout est dérivé
de la **seed** dans l'implémentation actuelle
(`random.Random(seed).shuffle`, `torch.Generator().manual_seed(seed)`) : la
seed EST le secret — elle doit être **aléatoire (CSPRNG)** et ne jamais être
publiée ni réutilisée.

### Voie A — re-transform sur Modal (rapide, posture expérimentale assumée)

La seed transite par Modal pendant la transformation (compromis documenté :
Modal = accélérateur expérimental ; la transformation est transférable en
local, voir Voie B).

```bash
SEED_NEW=$(openssl rand -n 4 -hex | head -c 8)   # 32 bits aléatoires, à défaut
# mieux : SEED_NEW=$(python3 -c "import secrets; print(secrets.randbits(64))")
~/modal-venv/bin/modal run modal_app.py::transform_chained \
    --seed "$SEED_NEW" --alpha-e 1 --alpha-h 0.2 --h 128 \
    --model-name /models/qwen3-8b-ft-gepa \
    --out-subdir qwen3-8b-ft-h128-a1-h02-r1
```

- modèle produit : `qwen3-8b-ft-h128-a1-h02-r1` sur le volume (α_e=1,0/α_h=0,2,
  le réglage défensif) ;
- la clé écrite sur `obfuscator-keys` (volume) : la télécharger puis **purger
  le volume** (posture, cf. docs/installation-demo.md §5) ;
- **retirer l'ancien modèle du volume** (multi-snapshot) et re-pointer
  `MODEL_SUBDIR` + `modal deploy modal_app.py`.

### Voie B — re-transform en local (la seed ne quitte pas la machine)

Nécessite ~48 Go de RAM (modèle clair + modèle reconstruit + temporaires
fp64). Appel direct de `obfuscate_chained` (pas de CLI dédiée — petit script,
mêmes paramètres que la cellule 27 du notebook) :

```python
import torch
from transformers import AutoModelForCausalLM
from aloepri.chained_transform import obfuscate_chained

seed = __import__("secrets").randbits(64)     # secret, jamais publié
clear = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", dtype=torch.bfloat16,
    attn_implementation="eager").eval()
obf, keys = obfuscate_chained(clear, clear.config, seed,
                              alpha_e=1.0, alpha_h=0.2, lam=0.3,
                              h=128, kappa_mode="empirical")
obf.to(torch.bfloat16).save_pretrained("qwen3-8b-ft-h128-a1-h02-r1")
import json
json.dump(keys, open("artifacts/obfuscation_keys.json", "w"),
          ensure_ascii=False)   # clé locale, jamais envoyée
```

> La transformation doit partir du modèle **fine-tuné** (`qwen3-8b-ft-gepa`,
> volume) pour reproduire le modèle défensif, pas de la base publique.

### Commun aux deux voies

1. **Avant** : noter l'état servi (`MODEL_SUBDIR`, modèle sur volume, clé
   client) pour pouvoir vérifier la bascule.
2. **Après** : vérifier `verify_chained` (corrélation logits + top-1) sur le
   nouveau modèle, round-trip client avec la nouvelle clé, `/health`.
3. **Purger** : ancien modèle du volume, ancienne clé locale, volume
   `obfuscator-keys` s'il a été utilisé (Voie A).
4. **Ne pas publier** la nouvelle seed dans le repo, le notebook ou STATUS.md
   (le notebook documente `seed 0` = reproductibilité expérimentale — une
   rotation réelle sort de ce cadre).

---

## Cycle recommandé (usage opérationnel simple)

- Clé de **démonstration** : générer une clé aléatoire (niveau 1) avant chaque
  démo publique ; `seed 0` uniquement pour les expériences reproductibles.
- Usage **confidentiel** (si un jour) : rotation complète (niveau 2) au
  démarrage, puis rotation périodique — fréquence à définir selon le volume
  d'échanges (le papier reconnaît les attaques fréquentielles TFMA/SDA : une
  permutation durable accumule des statistiques).

## Limites documentées (sans fard)

- Le niveau 1 ne protège pas contre le multi-snapshot ; le niveau 2 oui, mais
  sa résistance à des **observations croisées de plusieurs générations**
  complètes n'a pas été mesurée expérimentalement (axe de travail ouvert).
- Les **métadonnées** (longueur des requêtes, cadence, corrélations
  temporelles) ne sont pas masquées par AloePri — une rotation n'y change
  rien.
- La récupération résiduelle VMA (~8,35 % vue gate à α_e=1,0 ; 0 % vue
  W_e·W_h) et le canal hidden ISA (100 % des ids permutés) restent vrais à
  chaque génération : la rotation renouvelle le secret, pas ces canaux.
