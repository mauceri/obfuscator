---
marp: true
theme: default
paginate: true
header: "Confidentialité des échanges avec un LLM dans le cloud — démonstration"
footer: "Solution frugale, locale, confidentielle, et souple"
style: |
  section { font-size: 24px; }
  h1 { color: #1a3c6e; }
  h2 { color: #1a3c6e; }
  em.rouge { color: #a11; font-style: normal; font-weight: bold; }
---

# Confidentialité des échanges avec un LLM « dans le cloud »

**Le problème** : quand un cabinet envoie un texte à un grand modèle de
langage (LLM) hébergé chez un tiers, le prestataire **voit tout** — les
questions, les réponses, et les métadonnées (qui, quand, combien).

Pour des données couvertes par le **secret professionnel**, c'est rédhibitoire —
ou cela impose de tout héberger soi-même.

**La question** : peut-on utiliser la puissance d'un LLM distant **sans lui
montrer le texte** ?

---

# Les trois réponses classiques… et leurs limites

| Option | Confidentialité | Coût | Entretien |
|---|---|---|---|
| LLM public (ChatGPT, Claude…) | ❌ le fournisseur lit tout | faible | nul |
| LLM privé chez un hébergeur | ⚠️ contrat + confiance, le texte circule en clair chez lui | moyen | faible |
| LLM **local** (dans le cabinet) | ✅ rien ne sort | **élevé** : GPU dédié, électricité, expertise | **lourd** |

La solution locale est la seule vraiment confidentielle — mais elle est chère
et difficile à maintenir pour une petite structure.

**Notre proposition** : une **quatrième voie** — le texte sort de la machine,
**mais illisible** pour le serveur qui le traite.

---

# L'idée : chiffrer le texte… sans empêcher le modèle de répondre

Un LLM ne peut pas répondre à un texte chiffré classiquement (AES, TLS) : il
doit « lire » le sens.

**Mais on peut transformer le modèle ET les données ensemble**, de façon
cohérente :

- le texte est **permuté** : chaque mot est remplacé par un **numéro secret**
  (comme un code), selon une clé détenue **uniquement par le client** ;
- le modèle hébergé est **reconstruit** pour comprendre ces numéros, pas le
  texte ;
- les réponses reviennent sous forme de numéros — **le client les retraduit**.

Le serveur traite des nombres **sans signification pour lui**. C'est la
technique d'**obfuscation covariante** (AloePri, arXiv 2603.01499), que nous
avons implémentée et mesurée sur Qwen3-8B.

---

# Démonstration (1/2) — ce que voit le serveur

Le client envoie au serveur **uniquement des nombres** :

```
ids clairs (prompt)      : [2412, 16827, 4761, 100324, …]
ids PERMUTÉS envoyés     : [83410, 5611, 139432, 227, …]   ← ce que reçoit le serveur
ids PERMUTÉS reçus (rép.) : [9932, 44817, 716, …]
texte restitué            : « … Paris. »
```

- **Aucun mot, aucune phrase** ne transite en clair.
- La **clé de traduction** reste sur la machine du cabinet.
- Le serveur (et le transport) ne voient que des nombres.

---

# Démonstration (2/2) — le modèle servi

Le modèle hébergé est la version **transformée** du modèle public :

- ses poids ont été **reconstruits** (dimension interne étendue, matrices
  secrètes, bruit contrôlé) pour fonctionner **uniquement** avec la
  permutation du client ;
- il est **inutilisable** sans la clé : un tiers qui téléchargerait le modèle
  ne pourrait pas en tirer de texte ;
- le serveur ne détient **ni tokenizer privé, ni clé** : il ne peut pas
  retraduire les nombres qu'il manipule.

Résultat : on peut utiliser le LLM distant **sans lui confier le texte**.

---

# Ce que nous avons mesuré (sans fard)

Prototype Qwen3-8B (8 milliards de paramètres), réglage défensif complet :

- **Attaques de reconstruction des poids** (VMA, la plus redoutée) :
  - variante « embedding × gate » : **8 %** des correspondances seulement
    (le papier annonce 13-25 %) ;
  - variante « embedding × tête de sortie » (la plus profonde, table de
    46 Go) : **0 %** — rien n'est récupéré ;
- **Qualité conservée** : perplexité +13 % sur texte courant, réponse en
  question/réponse notée **4,4/5** par un juge indépendant (vs 4,6/5 sans
  protection).

**La protection a un coût mesuré, modeste — pas une dégradation invisible,
pas une perte.**

---

# Ce que cette approche ne protège PAS (transparence)

1. **Métadonnées** : le prestataire voit la longueur des échanges, leur
   fréquence, leurs heures. Pas le contenu.
2. **Analyses statistiques longues** : une même clé utilisée très longtemps
   peut laisser filtrer des régularités — d'où la **rotation régulière des
   clés** (procédure opérationnelle prévue).
3. **L'état interne du modèle** : un serveur malveillant très technique peut
   extraire les numéros traités — **mais sans la clé, ce sont des nombres**.
4. C'est un **prototype de recherche validé**, pas encore un produit
   industrialisé (le déploiement « clé en main » reste à construire).

**En clair** : la confidentialité repose sur une **clé locale** — comme pour
tout chiffrement. Celui qui a la clé lit tout ; sans elle, rien.

---

# Pourquoi c'est intéressant pour un cabinet (1/2)

**Frugalité** — pas de GPU à acheter ni à entretenir :
- le modèle vit dans le cloud à la demande (facturation à l'usage, arrêt
  automatique entre les requêtes) : **moins d'un dollar par heure**
  d'utilisation réelle, contre **plusieurs milliers d'euros** pour une
  machine locale équivalente ;
- le poste client est un simple ordinateur portable.

**Confidentialité** — le texte ne sort jamais en clair :
- compatible avec le **secret professionnel** : le prestataire ne peut pas
  lire les dossiers, même s'il le voulait ;
- la clé reste sous le contrôle exclusif du cabinet.

---

# Pourquoi c'est intéressant pour un cabinet (2/2)

**Maintenance** — le plus lourd reste chez le tiers :
- mises à jour du modèle, pannes, capacité : gérées par l'hébergeur ;
- côté cabinet : une clé, un petit script, rien d'autre.

**Portabilité et souplesse** — la solution s'adapte aux contraintes :
- modèle **ouvert** (Qwen), déployable chez **n'importe quel** hébergeur —
  pas de verrouillage fournisseur ;
- la partie sensible (transformation + clés) peut se faire **sur une machine
  locale** modeste (64 Go de RAM) ou dans le cloud, au choix ;
- évolutif : du petit usage ponctuel au volume important, sans changement
  d'architecture.

> **Solution frugale, locale, confidentielle, et souple.**

---

# Et concrètement, aujourd'hui ?

Ce qui **existe et fonctionne** (démontrable en direct) :

- le modèle Qwen3-8B obfusqué (schéma complet h>0) **servi dans le cloud**,
  interrogé par un **client chiffré** local — la démonstration de ce soir ;
- les mesures de sécurité et de qualité (transparentes, y compris les
  limites) ;
- une **procédure de rotation des clés** documentée (deux niveaux : secours
  et complète).

Ce qui **reste à faire** avant une mise en service confidentielle :

- industrialisation (installation guidée, gestion des clés, surveillance) ;
- mesure des attaques croisées entre plusieurs générations de clés ;
- réduction des métadonnées visibles.

---

# Le mot de la fin

**Le vrai sujet n'est pas « faire confiance ou ne pas faire confiance » au
fournisseur de LLM — c'est de ne pas avoir à lui confier le texte.**

Notre démonstration montre une voie **frugale, locale, confidentielle et
souple** : la puissance d'un grand modèle distant, sans lui montrer les
dossiers.

**Questions ?**

---

<!-- Annexe technique minimale — à n'utiliser qu'en cas de question pointue -->

# Annexe — en une diapositive

- **Mécanisme** : permutation du vocabulaire (clé Π, client) + reconstruction
  du modèle (hidden d → d+2h, matrices clés P̂/Q̂, bruit α_e=1,0/α_h=0,2).
- **Référence** : AloePri — arXiv 2603.01499 (obfuscation covariante).
- **Modèle** : Qwen3-8B (ouvert, 8 Md de paramètres), fine-tuné sur corpus
  français puis obfusqué ; servi sur GPU cloud à la demande.
- **Mesures clés** : VMA gate 8 % (papier : 13-25 %) ; VMA embedding×head
  **0 %** ; qualité Q&A −4,3 % ; les **ids permutés** sont récupérables à
  100 % par un serveur technique (ISA) — **sans la clé, ce sont des
  nombres** ; métadonnées non masquées.
- **Séparation des rôles** : la clé et la transformation sensible peuvent
  vivre **en local** ; le cloud ne reçoit que le modèle transformé et des
  nombres.
