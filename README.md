# obfuscator

Notebook de procédure AloePri (arXiv 2603.01499) pour Qwen3-8B, orchestrant
Modal pour les étapes lourdes. Dépôt autonome : zéro dépendance à Secretarius
(POC utilisé uniquement comme référence).

## Prérequis

- Python 3.12.
- Venv local :

  ```bash
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  ```

- CLI Modal (`~/modal-venv/bin/modal setup`) — uniquement pour les étapes
  lourdes sur Modal (transform GPU, export, service).

## Lancement du notebook

```bash
.venv/bin/jupyter notebook
```

## Posture de sécurité

Les clés d'obfuscation (`obfuscation_keys*.json`) ne quittent **jamais** le
client : elles ne sont ni poussées (voir `.gitignore`) ni transmises à Modal.

## Coûts Modal

- Transform CPU : ~30-60 min
- A100-40GB : ~1,5-2 $/h
- L4 : ~0,80 $/h

## Liens spec / plan

- [Spec](docs/superpowers/specs/2026-08-24-aloepri-notebook-design.md)
- [Plan](docs/superpowers/plans/2026-08-24-aloepri-notebook.md)
