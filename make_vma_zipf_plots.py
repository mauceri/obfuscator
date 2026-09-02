"""Génère les deux images :
1. zipf_gepa.png     — diagramme de Zipf du corpus GEPA (rang × fréquence)
2. classes_recup.png — où tombent les tokens récupérés par la VMA

Usage : python make_vma_zipf_plots.py <tokens_recuperes.json> [n_testes]
<tokens_recuperes.json> : liste d'ids clairs récupérés (RESULTAT_VMA_FULL).
[n_testes] : taille de l'échantillon testé par la VMA (défaut 2000).
"""
import json, pickle, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- fréquences GEPA (mêmes données que la cellule de préparation) ---
data = pickle.load(open('/tmp/gepa_freq.pkl', 'rb'))
freq = data["freq"]; total = data["total"]
V_MODEL = data["V"]
items = sorted(freq.items(), key=lambda kv: -kv[1])
V_obs = len(items)
rank_of = {i: r for r, (i, _) in enumerate(items, start=1)}

# --- image 1 : Zipf ---
ranks = list(range(1, V_obs + 1))
pcts = [100 * c / total for _, c in items]
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(ranks, pcts, '.', markersize=3, alpha=0.7)
ax.set_xlabel("rang (1 = token le plus fréquent)")
ax.set_ylabel("fréquence (% du corpus)")
ax.set_title("Distribution de Zipf — corpus GEPA\n"
             f"{total:,} tokens, {V_obs} types (V modèle = {V_MODEL})")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig("zipf_gepa.png", dpi=150)
plt.close(fig)
print("image 1 : zipf_gepa.png")

# --- classes décimales de rang (définies sur TOUS les tokens GEPA) ---
def decile(rank):
    if rank <= 10: return "1-10"
    if rank <= 100: return "11-100"
    if rank <= 1000: return "101-1000"
    if rank <= 10000: return "1001-10000"
    return "10001+"

CLASSES = ["1-10", "11-100", "101-1000", "1001-10000", "10001+"]
recup_ids = json.load(open(sys.argv[1]))
n_testes = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
recup = set(recup_ids)

# taille de classe = nb de tokens GEPA dans la classe (dénominateur du %)
cls_size = {c: 0 for c in CLASSES}
for i, (tid, _) in enumerate(items, start=1):
    cls_size[decile(i)] += 1

# récupérés par classe (parmi les tokens GEPA récupérés)
rec_by_cls = {c: 0 for c in CLASSES}
for tid in recup:
    r = rank_of.get(tid)
    if r:
        rec_by_cls[decile(r)] += 1

print(f"échantillon testé : {n_testes} | tokens récupérés : {len(recup)} "
      f"({100*len(recup)/n_testes:.2f}%)")
print(f"{'classe':<12} {'taille':>6} {'récup.':>7} {'% de la classe':>13}")
for c in CLASSES:
    s, r = cls_size[c], rec_by_cls[c]
    print(f"{c:<12} {s:>6} {r:>7} {100*r/s if s else float('nan'):>12.2f}%")

# --- image 2 : classes décimales, largeur ∝ taille, hauteur = % récupéré ---
fig, ax = plt.subplots(figsize=(9, 5))
widths = [max(cls_size[c], 1) for c in CLASSES]
heights = [100 * rec_by_cls[c] / cls_size[c] if cls_size[c] else 0
           for c in CLASSES]
# largeur ∝ racine carrée de la taille (visuel, sinon 1-10 invisible)
w_norm = [w ** 0.5 for w in widths]
x = range(len(CLASSES))
ax.bar(x, heights, width=[0.8 * w / max(w_norm) for w in w_norm],
       color=["#d62728" if h > 0 else "#cccccc" for h in heights],
       edgecolor="black", linewidth=0.5)
ax.set_xticks(list(x))
ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_ylabel("% de tokens récupérés dans la classe")
ax.set_xlabel("classe de rang (fréquence GEPA) — largeur ∝ taille de classe")
ax.set_title(f"Tokens récupérés par la VMA : {len(recup)} sur {n_testes} "
             f"testés ({100*len(recup)/n_testes:.1f} %)\n"
             "répartition par classe de fréquence (corpus GEPA)")
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig("classes_recup.png", dpi=150)
plt.close(fig)
print("image 2 : classes_recup.png")
