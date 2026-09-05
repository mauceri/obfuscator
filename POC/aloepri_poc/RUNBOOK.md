# RUNBOOK — déploiement RunPod, round-trip, mesures qualité/vitesse

> **Portage vers un autre modèle (Qwen3-8B, etc.)** : ce RUNBOOK décrit le
> déroulé historique sur Qwen2.5-7B-Instruct (RunPod). Avant de transformer
> un autre modèle, lancer `check_arch.py` (vérifie les hypothèses : GQA,
> head_dim, biais, q_norm/k_norm, vocabulaire) puis utiliser
> `transform_streaming.py` (variante mémoire-léger, ~5 Go de pic, produit
> bit-à-bit les mêmes poids) — cf. `aloepri_modal/README.md` pour le cas
> Qwen3-8B servi sur Modal. Qwen3 a des normes de tête q_norm/k_norm :
> `rope_scaling=off` y est automatique (le scaling diagonal Ĥ du papier ne
> commute pas avec une RMSNorm de tête).

Étapes manuelles de la Task 9 : louer un GPU, transformer le vrai
Qwen2.5-7B-Instruct, vérifier le round-trip de bout en bout, mesurer la
qualité et la vitesse de l'obfuscation, consigner les résultats. Le code
(`server.py`, `measure_quality.py`, `measure_speed.py`) est écrit et testé
(sans GPU) ; ce document est ce qu'il reste à exécuter à la main.

## Coût

RunPod Community Cloud, RTX A5000 (24 Go VRAM) ≈ **0,16 $/h** au moment de
l'écriture (vérifier le tarif courant sur RunPod avant de louer). Budget
prévu : quelques heures GPU pour l'ensemble de la session (transform +
round-trip + mesures), donc de l'ordre de **1 à 3 $**. Chaque étape ci-dessous
qui suppose le Pod loué et actif est marquée **[$]** ; tout le reste (lecture,
écriture de code, consignation) ne coûte rien et peut se faire hors ligne.

**Le Pod facture au temps écoulé, pas à l'usage du GPU.** `model_transform.py`
tourne entièrement sur CPU (aucun `.cuda()`, cf. sa docstring) — mais lancé sur
le Pod loué, chaque exécution consomme quand même des minutes facturées.
Minimiser le nombre d'allers-retours et de runs inutiles est donc une vraie
économie, pas un détail de style.

## 0. Prérequis

- Un compte RunPod avec de quoi payer (carte, ou crédits).
- Ce répertoire (`aloepri_poc/`) à jour et testé en local : `python3 -m
  pytest aloepri_poc/tests -q` doit être vert AVANT de louer quoi que ce soit.
- Accès à `Qwen/Qwen2.5-7B-Instruct` sur le Hub HuggingFace (public, pas
  d'accord de licence particulier au moment de l'écriture — vérifier sur la
  fiche du modèle).

## 1. Louer le Pod **[$]**

Sur RunPod : créer un Pod GPU **RTX A5000**, **Community Cloud**, image
PyTorch standard (CUDA + PyTorch préinstallés). Un seul GPU suffit ici — les
scripts de mesure chargent un modèle 7B à la fois (~14 Go en bf16), jamais les
deux simultanément (cf. §5).

Noter l'heure de démarrage pour le suivi du coût.

## 2. Transférer le code et installer les dépendances **[$]**

Depuis sanroque (remplacer `<POD_IP>` par l'IP/port SSH donnés par RunPod) :

```bash
rsync -avz ~/Secretarius/aloepri_poc/ root@<POD_IP>:/workspace/aloepri_poc/
ssh root@<POD_IP> "pip install -r /workspace/aloepri_poc/requirements.txt"
```

Le fichier `obfuscation_keys.json` n'existe pas encore à ce stade (il est
produit par `model_transform.py` à l'étape 3) — rien à exclure du rsync ici.

## 3. Vérifier les hypothèses (GQA, weight tying) **[$]**

```bash
ssh root@<POD_IP>
cd /workspace/aloepri_poc
python3 -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print(c.num_attention_heads, c.num_key_value_heads, hasattr(c, 'num_key_value_heads'))"
```

Attendu : confirme GQA (`num_key_value_heads` défini et < `num_attention_heads`).
Si absent, ou égal (MHA), ou qu'une architecture MLA est détectée : **arrêter**
et réévaluer Task 7 avant de continuer — `obfuscate_model_in_place` lève une
`AssertionError` explicite dans ce cas, donc l'étape 4 échouerait de toute
façon, mais mieux vaut le savoir avant de lancer une transformation de 15 Go.

## 4. Transformer le modèle — run principal + runs de contrôle **[$]**

`model_transform.py` expose déjà tous les leviers nécessaires en CLI
(`--alpha-e`, `--alpha-h`, `--beta`, `--zeta`, `--seed`) — pas de nouveau code
à écrire ici, seulement plusieurs invocations avec des `--output`/`--keys`
distincts pour ne pas s'écraser les uns les autres.

### 4a. Run principal (défauts du plan)

```bash
python3 model_transform.py --model Qwen/Qwen2.5-7B-Instruct \
    --output ./obfuscated_model --keys ./keys_default.json --seed 0
```

### 4b. Run de contrôle β=1 (Ẑ_block = identité)

**Pourquoi.** Ẑ_block est la seule approximation non exacte du pipeline
d'attention ; elle coûte **1,4 % à 6 %** d'erreur relative sur les scores
d'attention avec ζ=1e3 (défaut), et **12 % à 35 %** avec ζ=rope_theta=1e6.
β=1 force Ẑ_block = identité (plus de mélange de fréquences RoPE) : un run à
β=1 sépare donc « la reparamétrisation est-elle correcte ? » de « combien
coûte l'approximation Ẑ_block ? ». Si la qualité (§6) est mauvaise sur le run
principal (4a) mais bonne à β=1, la cause est isolée : Ẑ_block, pas un bug de
reparamétrisation.

```bash
python3 model_transform.py --model Qwen/Qwen2.5-7B-Instruct \
    --output ./obfuscated_model_beta1 --keys ./keys_beta1.json --seed 0 --beta 1
```

### 4c. Point de balayage α_e

**Pourquoi.** Le bruit d'embedding (α_e, α_h) est la seule dégradation NON
compensée du POC (cf. docstring de `model_transform.py`) — à α_e=1.0 (défaut
du papier), le bruit a la même dispersion que le poids lui-même. Un point de
balayage à α_e plus faible donne un deuxième point de comparaison pour situer
où se place le compromis qualité/confidentialité de ce paramètre précis.

```bash
python3 model_transform.py --model Qwen/Qwen2.5-7B-Instruct \
    --output ./obfuscated_model_alpha05 --keys ./keys_alpha05.json --seed 0 --alpha-e 0.5
```

### 4d. (Optionnel, coûte un run de plus) ζ=rope_theta

Si le run principal (4a) montre une dégradation de qualité gênante, ζ est le
levier à essayer avant tout autre : le défaut du plan (ζ=1e3) NE coïncide PAS
avec le `rope_theta` réel de Qwen2.5-7B (1e6), et Task 8 a mesuré que ce choix
change l'erreur de Ẑ_block d'un facteur ~6 à 8× (12-35 % à ζ=1e6 vs 1,4-6 % à
ζ=1e3 par défaut — donc le défaut est le SENS attendu, mais l'arbitrage
mérite d'être vérifié empiriquement plutôt que supposé) :

```bash
python3 model_transform.py --model Qwen/Qwen2.5-7B-Instruct \
    --output ./obfuscated_model_zeta1e6 --keys ./keys_zeta1e6.json --seed 0 --zeta 1e6
```

Ne pas lancer ce run par défaut — seulement si 4a déçoit. Chaque run
supplémentaire consomme du temps Pod facturé (cf. §Coût).

## 5. Round-trip de bout en bout (critère de succès n°1 de la spec) **[$]**

Sur le run PRINCIPAL (4a) uniquement — c'est un test de correction, pas de
qualité, donc un seul run suffit.

```bash
# Lancer le serveur en arrière-plan
python3 server.py --model-dir ./obfuscated_model &

python3 -c "
from client_wrapper import ClientCodec
from transformers import AutoTokenizer
import json, requests

with open('keys_default.json') as f:
    keys = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
codec = ClientCodec(keys['vocab_permutation'], keys['vocab_unpermute'], tokenizer)

prompt = 'Quelle est la capitale de la France ?'
permuted_ids = codec.encode(prompt)
resp = requests.post('http://localhost:8000/generate', json={'input_ids': permuted_ids, 'max_new_tokens': 50})
output_permuted = resp.json()['output_ids']
print(codec.decode(output_permuted))
"
```

Vérifier MANUELLEMENT que la sortie est un texte français cohérent. C'est le
critère de succès n°1 de la spec — pas de script pour l'automatiser, une
lecture humaine du texte produit.

Arrêter le serveur (`fg` puis Ctrl-C, ou `kill %1`) avant de passer à la
mesure — `measure_quality.py`/`measure_speed.py` chargent leurs propres
modèles directement, ils n'ont pas besoin du serveur HTTP.

## 6. Mesure qualité — run principal + runs de contrôle **[$]**

**Point de sécurité à connaître avant de lancer ceci** : contrairement à
`server.py` (qui ne charge jamais `obfuscation_keys.json`), `measure_quality.py`
et `measure_speed.py` en ont besoin LOCALEMENT — donc sur le Pod, le temps de
la mesure — pour permuter correctement les IDs envoyés au modèle obfusqué
(sans quoi la perplexité mesurée serait celle d'un modèle nourri aux mauvaises
lignes d'embedding, pas celle de l'obfuscation réelle). C'est une dérogation
volontaire et limitée à ce script de benchmarking, où le développeur contrôle
le Pod loué de bout en bout ; ce n'est PAS la posture de `server.py` en
service, qui ne voit jamais les clés (cf. §8, nettoyage).

Une invocation par run produit du §4 (répéter `--obfuscated`/`--keys`) :

```bash
python3 measure_quality.py --baseline Qwen/Qwen2.5-7B-Instruct \
    --obfuscated ./obfuscated_model --keys ./keys_default.json \
    | tee resultats_quality_default.txt

python3 measure_quality.py --baseline Qwen/Qwen2.5-7B-Instruct \
    --obfuscated ./obfuscated_model_beta1 --keys ./keys_beta1.json \
    | tee resultats_quality_beta1.txt

python3 measure_quality.py --baseline Qwen/Qwen2.5-7B-Instruct \
    --obfuscated ./obfuscated_model_alpha05 --keys ./keys_alpha05.json \
    | tee resultats_quality_alpha05.txt
```

(+ `resultats_quality_zeta1e6.txt` si le run 4d a été fait.)

Chaque invocation charge baseline et obfuscated l'un après l'autre, jamais
simultanément (cf. `measure_quality.py`, commentaire sur les 24 Go de VRAM
d'une A5000) — pas d'action requise ici, c'est déjà géré par le script.

## 7. Mesure vitesse — run principal uniquement **[$]**

Contrairement à la qualité, le coût en tokens/s ne dépend pas de α_e/β/ζ (ce
sont des scalaires appliqués aux poids, la forme du calcul ne change pas) : un
seul run suffit, sur le modèle principal (4a).

```bash
python3 measure_speed.py --baseline Qwen/Qwen2.5-7B-Instruct \
    --obfuscated ./obfuscated_model --keys ./keys_default.json \
    | tee resultats_speed.txt
```

## 8. Nettoyer les clés, arrêter le Pod

**Avant** d'arrêter le Pod : supprimer toute copie de `keys_*.json` qui s'y
trouve encore (elles n'y ont été copiées que pour la mesure, §6 — cf.
`model_transform.py` : « keys_path reste côté client — c'est le secret du
POC, il ne doit jamais être copié sur le Pod » en régime normal).

```bash
ssh root@<POD_IP> "rm -f /workspace/aloepri_poc/keys_*.json"
```

Puis, depuis RunPod (interface web ou CLI) : **arrêter/terminer le Pod** pour
ne plus être facturé. Noter l'heure d'arrêt et le coût effectif affiché par
RunPod pour §9.

## 9. Consigner les résultats

Créer `aloepri_poc/RESULTATS.md` (nouveau fichier, hors scope de cette tâche
de rédaction de code — à écrire quand les vraies mesures existent) avec, au
minimum :

- Un tableau delta qualité par prompt pour CHAQUE run (4a, 4b, 4c, et 4d si
  fait), plus la moyenne imprimée en bas de la sortie de `measure_quality.py`.
- La sortie de `measure_speed.py` (tok/s et latence, baseline vs obfusqué,
  overhead en %).
- Le coût Pod effectif (durée totale × tarif RunPod affiché).
- Une phrase de conclusion explicite reliant le delta qualité du run 4a à ce
  qu'en expliquent 4b/4c : par exemple « la dégradation vient majoritairement
  de α_e (comparer 4a à 4c) » ou « Ẑ_block domine (comparer 4a à 4b) ».

```bash
cd ~/Secretarius   # ou le worktree courant
git add aloepri_poc/RESULTATS.md
git commit -m "docs(aloepri): résultats qualité/vitesse du POC sur RunPod"
```

Ne jamais ajouter `keys_*.json`/`obfuscation_keys.json` à ce commit — ce sont
des artefacts client, pas du code, et leur contenu (la permutation de
vocabulaire) est le secret que tout le POC vise à protéger.
