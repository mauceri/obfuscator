---
marp: true
theme: default
paginate: true
header: "Un assistant de cabinet dont le cerveau distant ne lit jamais vos dossiers — démonstration"
footer: "Solution frugale, locale, confidentielle, et souple"
style: |
  section { font-size: 24px; }
  h1 { color: #1a3c6e; }
  h2 { color: #1a3c6e; }
  em.rouge { color: #a11; font-style: normal; font-weight: bold; }
---

# Un assistant qui connaît vos dossiers… sans que personne dehors ne les lise

**Le problème** : un assistant utile à un cabinet doit connaître les dossiers,
les notes, les pièces. Or les grands modèles de langage (LLM) capables de les
exploiter vivent chez des tiers, et **le prestataire voit tout** : questions,
réponses, extraits de dossiers, métadonnées.

Pour des données couvertes par le **secret professionnel**, c'est rédhibitoire —
ou cela impose de tout héberger soi-même.

**La question** : peut-on avoir un assistant qui travaille sur les dossiers du
cabinet, avec la puissance d'un grand modèle distant, **sans jamais lui montrer
le texte** ?

---

# Les trois réponses classiques… et leurs limites

| Option | Confidentialité | Coût | Entretien |
|---|---|---|---|
| Assistant sur LLM public (ChatGPT, Claude…) | ❌ le fournisseur lit tout | faible | nul |
| LLM privé chez un hébergeur | ⚠️ contrat + confiance, le texte circule en clair chez lui | moyen | faible |
| Tout **local** (dans le cabinet) | ✅ rien ne sort | **élevé** : GPU dédié, électricité, expertise | **lourd** |

La solution locale est la seule vraiment confidentielle — mais elle est chère,
et un petit modèle local reste limité.

**Notre proposition** : une **quatrième voie** — l'assistant, ses outils et vos
dossiers restent au cabinet ; **seul le cerveau est distant**, et il ne
reçoit que des nombres **illisibles** pour qui n'a pas la clé.

---

# L'assistant : ce qui reste au cabinet, ce qui part

**Au cabinet** (un simple ordinateur, pas de GPU) :

- l'**interface** : une messagerie (Telegram) ou l'éditeur de notes (Obsidian) ;
- la **base de connaissances** : dossiers, notes, pièces capturées et indexées
  sur place ;
- les **outils** de l'assistant (recherche dans les dossiers, lecture de
  pièces, rédaction) exécutés localement, dans un bac à sable ;
- la **clé** de traduction.

**Dans le cloud, à la demande** : uniquement le modèle de langage, dans une
version **transformée**, qui ne reçoit et ne renvoie que des **nombres**.

Rien ne sort du cabinet en clair : ni les questions, ni les extraits de
dossiers, ni les réponses.

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
avons implémentée et mesurée sur Qwen3-8B, et que nous portons sur Qwen3-14B.

---

# Démonstration (1/2) — une question sur un dossier

L'avocat écrit à l'assistant, depuis Telegram :

> `/q` clause de non-concurrence dans le dossier Dupont

1. L'assistant **cherche au cabinet**, dans la base de connaissances : il
   retrouve les passages pertinents (rien n'a quitté la machine) ;
2. question + passages sont **traduits en numéros** avec la clé locale ;
3. le modèle distant reçoit **uniquement des nombres** et renvoie des nombres ;
4. l'assistant **retraduit** et répond à l'avocat.

Ce que reçoit le serveur :

```
ids clairs (question + passages) : [2412, 16827, 4761, 100324, …]
ids PERMUTÉS envoyés             : [83410, 5611, 139432, 227, …]   ← ce que voit le serveur
ids PERMUTÉS reçus (réponse)     : [9932, 44817, 716, …]
texte restitué au cabinet        : « La clause prévoit … »
```

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

Résultat : l'assistant profite d'un grand modèle distant **sans lui confier
un seul mot** des dossiers.

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

Le passage à Qwen3-14B (en cours) sera mesuré de la même façon, y compris la
fiabilité des **appels d'outils** de l'assistant sous obfuscation.

**La protection a un coût mesuré, modeste — pas une dégradation invisible,
pas une perte.**

---

# Ce que cette approche ne protège PAS (transparence)

1. **Métadonnées** : le prestataire voit la longueur des échanges (donc la
   taille des extraits envoyés), leur fréquence, leurs heures. Pas le contenu.
2. **Analyses statistiques longues** : une même clé utilisée très longtemps
   peut laisser filtrer des régularités — d'où la **rotation régulière des
   clés** (procédure opérationnelle documentée).
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
- la recherche dans les dossiers se fait **au cabinet** ; seuls les extraits
  utiles partent, **en nombres** ;
- compatible avec le **secret professionnel** : le prestataire ne peut pas
  lire les dossiers, même s'il le voulait ;
- la clé reste sous le contrôle exclusif du cabinet.

---

# Pourquoi c'est intéressant pour un cabinet (2/2)

**Maintenance** — le plus lourd reste chez le tiers :
- mises à jour du modèle, pannes, capacité : gérées par l'hébergeur ;
- côté cabinet : une clé, l'assistant, rien d'autre.

**Portabilité et souplesse** — la solution s'adapte aux contraintes :
- modèle **ouvert** (Qwen), déployable chez **n'importe quel** hébergeur —
  pas de verrouillage fournisseur ;
- le **cerveau est interchangeable** : le même assistant peut tourner sur un
  petit modèle local pour les tâches simples et sur le grand modèle distant
  obfusqué pour les tâches exigeantes, sans changer d'outils ni d'habitudes ;
- la partie sensible (transformation + clés) peut se faire **sur une machine
  locale** modeste (64 Go de RAM) ou dans le cloud, au choix.

> **Solution frugale, locale, confidentielle, et souple.**

---

# Et concrètement, aujourd'hui ?

Ce qui **existe et fonctionne** (démontrable en direct) :

- l'**assistant** au cabinet : messagerie Telegram, base de connaissances sur
  Obsidian, recherche et commandes déterministes, outils en bac à sable ;
- le modèle Qwen3-8B obfusqué (schéma complet h>0) **servi dans le cloud**,
  interrogé par un **client chiffré** local ;
- les mesures de sécurité et de qualité (transparentes, y compris les
  limites) ;
- une **procédure de rotation des clés** documentée (deux niveaux : secours
  et complète).

Ce qui est **en cours** : l'obfuscation de Qwen3-14B, puis le branchement du
cerveau obfusqué sur l'assistant, avec mesure des appels d'outils.

Ce qui **reste à faire** avant une mise en service confidentielle :
industrialisation (installation guidée, gestion des clés, surveillance),
mesure des attaques croisées entre générations de clés, réduction des
métadonnées visibles.

---

# Le mot de la fin

**Le vrai sujet n'est pas « faire confiance ou ne pas faire confiance » au
fournisseur de LLM — c'est de ne pas avoir à lui confier le texte.**

Notre démonstration montre une voie **frugale, locale, confidentielle et
souple** : un assistant qui travaille sur vos dossiers, au cabinet, avec la
puissance d'un grand modèle distant qui ne les lit jamais.

**Questions ?**

---

<!-- Annexe technique minimale — à n'utiliser qu'en cas de question pointue -->

# Annexe — en une diapositive

- **Assistant** : Secretarius (OpenClaw) — routeur local phi-4-mini pour les
  commandes, agents dédiés par fonction (base de connaissances, courrier,
  lecture web isolée), outils exécutés en bac à sable ; le cerveau de chaque
  agent est un fournisseur interchangeable (local, cloud, cloud obfusqué).
- **Mécanisme** : permutation du vocabulaire (clé Π, client) + reconstruction
  du modèle (hidden d → d+2h, matrices clés P̂/Q̂, bruit α_e=1,0/α_h=0,2).
- **Référence** : AloePri — arXiv 2603.01499 (obfuscation covariante).
- **Modèle** : Qwen3-8B (ouvert), fine-tuné sur corpus français puis
  obfusqué ; servi sur GPU cloud à la demande. Qwen3-14B en cours.
- **Mesures clés (8B)** : VMA gate 8 % (papier : 13-25 %) ; VMA
  embedding×head **0 %** ; qualité Q&A −4,3 % ; les **ids permutés** sont
  récupérables à 100 % par un serveur technique (ISA) — **sans la clé, ce
  sont des nombres** ; métadonnées non masquées.
- **Séparation des rôles** : la clé, les dossiers, les outils et la
  transformation sensible vivent **au cabinet** ; le cloud ne reçoit que le
  modèle transformé et des nombres.
