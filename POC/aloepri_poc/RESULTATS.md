# Résultats — POC AloePri sur Qwen2.5-7B-Instruct (RunPod)

Mesures réelles produites en exécutant `RUNBOOK.md` de bout en bout sur un
Pod RunPod (GPU RTX A5000, 40 Go de volume réseau, région EU-FR-1).
Aucune donnée fabriquée — voir `RUNBOOK.md` pour la procédure exacte.

- **Modèle** : `Qwen/Qwen2.5-7B-Instruct`, bfloat16
- **Seed** : 0
- **Date** : 2026-08-19
- **Périmètre** : h=0 (pas de matrices clés P̂/Q̂), pas de chaînage
  inter-couches (cf. `docs/superpowers/specs/2026-08-17-aloepri-poc-complet-design.md`)

## 1. Round-trip qualitatif (4a)

Prompt : « Quelle est la capitale de la France ? »

Sortie décodée (greedy, 50 tokens) :

> La capitale de la France est Paris.
>
> Voici quelques informations supplémentaires sur Paris :
>
> 1. C'est la plus grande ville de France en termes de population.
> 2. Elle est située dans le nord-ouest du pays [...]

Cohérente, correcte, en français — le pipeline complet (permutation
vocabulaire + bruit + FFN + attention obfusquée) round-trip proprement.

## 2. Résultats qualité (perplexité, delta % vs baseline, 24 prompts factuels courts)

### 4a — run principal (β=8, α_e=1.0, α_h=0.2, λ=0.3, ζ=1e3 — défauts du plan)

| Prompt | baseline | obfusqué | delta |
|---|---|---|---|
| La capitale de la France est | 18.09 | 22.02 | +21.7% |
| Le théorème de Pythagore énonce que | 6.81 | 8.83 | +29.6% |
| En 1789, la Révolution française | 4.19 | 7.59 | +81.0% |
| L'eau bout à une température de | 42.73 | 59.76 | +39.9% |
| La photosynthèse est le processus par le[quel] | 4.17 | 4.04 | -3.3% |
| Albert Einstein a formulé la théorie de | 17.96 | 21.49 | +19.7% |
| Le plus grand océan du monde est | 9.96 | 8.02 | -19.5% |
| La Terre tourne autour du Soleil en environ | 11.55 | 9.26 | -19.8% |
| Le symbole chimique de l'or est | 14.17 | 14.78 | +4.3% |
| La tour Eiffel a été construite en | 5.12 | 5.34 | +4.4% |
| Un triangle équilatéral possède | 7.51 | 7.99 | +6.4% |
| La vitesse de la lumière dans le vide est d'environ | 4.35 | 5.61 | +28.9% |
| Le Sahara est le plus grand désert | 13.59 | 12.74 | -6.2% |
| Napoléon Bonaparte est devenu empereur en | 5.48 | 8.35 | +52.3% |
| La monnaie officielle du Japon est | 10.77 | 12.60 | +17.0% |
| Le fleuve le plus long d'Afrique est | 7.05 | 6.67 | -5.3% |
| La Joconde a été peinte par | 14.56 | 25.92 | +78.0% |
| Un octogone possède | 23.43 | 25.99 | +10.9% |
| La capitale de l'Italie est | 13.87 | 21.78 | +57.1% |
| Le corps humain contient environ | 15.47 | 14.62 | -5.5% |
| La Seconde Guerre mondiale s'est terminée en | 4.43 | 7.33 | +65.5% |
| Le plus haut sommet du monde est | 13.75 | 9.13 | -33.6% |
| L'ADN est une molécule qui | 8.51 | 9.99 | +17.4% |
| La devise de la République française est | 9.59 | 11.19 | +16.7% |

**Moyenne sur 24 prompts : delta = +19.1%**

### 4b — contrôle β=1 (Ẑ_block = identité, isole le coût de l'approximation BlockPerm)

| Prompt | baseline | obfusqué | delta |
|---|---|---|---|
| La capitale de la France est | 18.09 | 21.89 | +21.0% |
| Le théorème de Pythagore énonce que | 6.81 | 8.82 | +29.5% |
| En 1789, la Révolution française | 4.19 | 7.51 | +79.2% |
| L'eau bout à une température de | 42.73 | 61.25 | +43.3% |
| La photosynthèse est le processus par le[quel] | 4.17 | 4.06 | -2.7% |
| Albert Einstein a formulé la théorie de | 17.96 | 21.43 | +19.3% |
| Le plus grand océan du monde est | 9.96 | 8.09 | -18.8% |
| La Terre tourne autour du Soleil en environ | 11.55 | 9.27 | -19.8% |
| Le symbole chimique de l'or est | 14.17 | 14.70 | +3.7% |
| La tour Eiffel a été construite en | 5.12 | 5.39 | +5.2% |
| Un triangle équilatéral possède | 7.51 | 8.01 | +6.8% |
| La vitesse de la lumière dans le vide est d'environ | 4.35 | 5.61 | +28.9% |
| Le Sahara est le plus grand désert | 13.59 | 12.70 | -6.5% |
| Napoléon Bonaparte est devenu empereur en | 5.48 | 8.35 | +52.4% |
| La monnaie officielle du Japon est | 10.77 | 12.47 | +15.8% |
| Le fleuve le plus long d'Afrique est | 7.05 | 6.68 | -5.2% |
| La Joconde a été peinte par | 14.56 | 25.63 | +76.1% |
| Un octogone possède | 23.43 | 24.77 | +5.7% |
| La capitale de l'Italie est | 13.87 | 21.68 | +56.3% |
| Le corps humain contient environ | 15.47 | 14.46 | -6.5% |
| La Seconde Guerre mondiale s'est terminée en | 4.43 | 7.40 | +67.0% |
| Le plus haut sommet du monde est | 13.75 | 9.20 | -33.1% |
| L'ADN est une molécule qui | 8.51 | 9.81 | +15.3% |
| La devise de la République française est | 9.59 | 11.18 | +16.6% |

**Moyenne sur 24 prompts : delta = +18.7%**

### 4c — balayage α_e=0.5 (bruit d'embedding réduit de moitié, β=8 par défaut)

| Prompt | baseline | obfusqué | delta |
|---|---|---|---|
| La capitale de la France est | 18.09 | 22.42 | +24.0% |
| Le théorème de Pythagore énonce que | 6.81 | 8.79 | +29.0% |
| En 1789, la Révolution française | 4.19 | 6.58 | +56.9% |
| L'eau bout à une température de | 42.73 | 46.33 | +8.4% |
| La photosynthèse est le processus par le[quel] | 4.17 | 3.99 | -4.5% |
| Albert Einstein a formulé la théorie de | 17.96 | 17.80 | -0.9% |
| Le plus grand océan du monde est | 9.96 | 8.87 | -10.9% |
| La Terre tourne autour du Soleil en environ | 11.55 | 9.25 | -19.9% |
| Le symbole chimique de l'or est | 14.17 | 16.01 | +13.0% |
| La tour Eiffel a été construite en | 5.12 | 5.05 | -1.3% |
| Un triangle équilatéral possède | 7.51 | 7.98 | +6.3% |
| La vitesse de la lumière dans le vide est d'environ | 4.35 | 5.70 | +31.1% |
| Le Sahara est le plus grand désert | 13.59 | 12.83 | -5.5% |
| Napoléon Bonaparte est devenu empereur en | 5.48 | 8.19 | +49.5% |
| La monnaie officielle du Japon est | 10.77 | 12.27 | +13.9% |
| Le fleuve le plus long d'Afrique est | 7.05 | 6.32 | -10.3% |
| La Joconde a été peinte par | 14.56 | 21.09 | +44.8% |
| Un octogone possède | 23.43 | 26.28 | +12.2% |
| La capitale de l'Italie est | 13.87 | 21.84 | +57.5% |
| Le corps humain contient environ | 15.47 | 15.95 | +3.1% |
| La Seconde Guerre mondiale s'est terminée en | 4.43 | 5.68 | +28.1% |
| Le plus haut sommet du monde est | 13.75 | 10.18 | -26.0% |
| L'ADN est une molécule qui | 8.51 | 9.74 | +14.5% |
| La devise de la République française est | 9.59 | 10.81 | +12.7% |

**Moyenne sur 24 prompts : delta = +13.6%**

### 4d — contrôle ζ=1e6 (rope_theta réel de Qwen2.5-7B, β=8/α_e=1.0 par défaut)

| Prompt | baseline | obfusqué | delta |
|---|---|---|---|
| La capitale de la France est | 18.09 | 21.92 | +21.2% |
| Le théorème de Pythagore énonce que | 6.81 | 8.75 | +28.4% |
| En 1789, la Révolution française | 4.19 | 7.39 | +76.2% |
| L'eau bout à une température de | 42.73 | 59.50 | +39.3% |
| La photosynthèse est le processus par le[quel] | 4.17 | 4.04 | -3.1% |
| Albert Einstein a formulé la théorie de | 17.96 | 21.40 | +19.1% |
| Le plus grand océan du monde est | 9.96 | 8.10 | -18.7% |
| La Terre tourne autour du Soleil en environ | 11.55 | 9.28 | -19.6% |
| Le symbole chimique de l'or est | 14.17 | 14.63 | +3.2% |
| La tour Eiffel a été construite en | 5.12 | 5.36 | +4.8% |
| Un triangle équilatéral possède | 7.51 | 7.97 | +6.2% |
| La vitesse de la lumière dans le vide est d'environ | 4.35 | 5.64 | +29.8% |
| Le Sahara est le plus grand désert | 13.59 | 12.76 | -6.1% |
| Napoléon Bonaparte est devenu empereur en | 5.48 | 8.30 | +51.4% |
| La monnaie officielle du Japon est | 10.77 | 12.55 | +16.6% |
| Le fleuve le plus long d'Afrique est | 7.05 | 6.72 | -4.6% |
| La Joconde a été peinte par | 14.56 | 25.86 | +77.6% |
| Un octogone possède | 23.43 | 25.41 | +8.5% |
| La capitale de l'Italie est | 13.87 | 21.73 | +56.7% |
| Le corps humain contient environ | 15.47 | 14.46 | -6.5% |
| La Seconde Guerre mondiale s'est terminée en | 4.43 | 7.40 | +67.1% |
| Le plus haut sommet du monde est | 13.75 | 9.07 | -34.1% |
| L'ADN est une molécule qui | 8.51 | 9.79 | +15.1% |
| La devise de la République française est | 9.59 | 11.16 | +16.4% |

**Moyenne sur 24 prompts : delta = +18.5%**

## 3. Résultats vitesse (run principal 4a uniquement)

Le coût en tokens/s ne dépend pas de α_e/β/ζ (scalaires appliqués aux
poids, la forme du calcul ne change pas) — un seul run suffit
(cf. `RUNBOOK.md` §7).

Prompt : « Décris en trois phrases le fonctionnement d'un transformer. »,
100 tokens générés, greedy.

| | tok/s | latence |
|---|---|---|
| baseline | 40.6 | 2.46s |
| obfusqué | 50.7 | 1.97s |

**Overhead vitesse : -24.8%** (l'obfusqué mesure *plus rapide* que le
baseline).

**Réserve méthodologique** : mesure sur un seul essai, chargement
séquentiel où le baseline est chargé et exécuté en premier. L'écart en
faveur de l'obfusqué est vraisemblablement un effet d'échauffement GPU
(warmup CUDA/kernels sur le premier appel de `generate()`) plutôt qu'un
gain réel attribuable à l'obfuscation elle-même. À ne pas lire comme
« l'obfuscation accélère l'inférence » sans un protocole plus rigoureux
(plusieurs essais, ordre alterné baseline/obfusqué). Ce que ce chiffre
établit solidement, en revanche, c'est l'absence de ralentissement notable
— l'obfuscation n'ajoute pas de surcoût de calcul structurel visible.

## 4. Coût Pod

**Coût effectif : 3,38 $** (dépense RunPod du jour, GPU RTX A5000 + volume
réseau 40 Go EU-FR-1, pour l'ensemble de la session : transformation du
modèle et mesures qualité/vitesse des 4 runs 4a-4d).

## 5. Conclusion

**4a vs 4b (β=1, Ẑ_block=identité)** : delta quasi identique (+19.1% vs
+18.7%) → **Ẑ_block n'est PAS la source principale de dégradation**,
contrairement à l'hypothèse de départ du RUNBOOK (l'approximation coûte
1,4–6% d'erreur sur les seuls scores d'attention à ζ=1e3 — un ordre de
grandeur en dessous du delta de perplexité observé de bout en bout).

**4a vs 4d (ζ=1e6, rope_theta réel de Qwen)** : delta quasi identique
également (+19.1% vs +18.5%) → faire coïncider ζ avec le `rope_theta` réel
du modèle **n'aggrave pas** la qualité de bout en bout, malgré une erreur
Ẑ_block 6–8× plus élevée en théorie sur les scores d'attention seuls. Le
levier ζ ne s'est pas révélé utile empiriquement dans ce POC.

**4a vs 4c (α_e=0.5)** : delta réduit significativement (+19.1% → +13.6%,
soit environ -29% relatif) → **le bruit d'embedding (α_e) est le
contributeur identifié le plus net** à la dégradation de qualité parmi les
leviers testés, bien qu'il n'explique pas la totalité du delta (13.6%
résiduel subsiste même à α_e réduit de moitié).

**Conclusion générale** : sur ce POC (Qwen2.5-7B-Instruct, h=0, sans
chaînage inter-couches, schéma complet embedding+FFN+attention),
l'obfuscation coûte en moyenne **+19% de perplexité** sur 24 prompts
factuels courts, avec une forte variance par prompt (de -34% à +81%). Le
round-trip reste qualitativement cohérent et correct malgré cette
dégradation quantitative. Parmi les paramètres testés, **α_e est le
levier de réglage qualité le plus efficace** ; β (Ẑ_block) et ζ n'ont
montré aucun effet mesurable dans la plage testée — un résultat qui
contredit l'hypothèse de risque n°1 posée dans le RUNBOOK avant mesure, et
qui aurait justifié d'explorer un balayage α_e plus poussé (0.3, 0.2...)
si l'objectif avait été de minimiser la dégradation à confidentialité
constante plutôt que de conclure ce POC de décision.
