# Plots

Ce dossier contient les scripts et les données utilisés pour générer les graphiques du rapport.

## Fichiers

- `generate_stage1_plots.py` : génère les fichiers de données CSV et les snippets LaTeX/TikZ utilisés dans le rapport.
- `initial_render_update.csv` : données de la mesure interactive initiale.
- `stage1_scaling.csv` : données de benchmark de l'étape 1.
- `initial_render_update_plot.tex` : graphique en barres de la mesure initiale.
- `stage1_speedup_plot.tex` : courbe de speed-up.
- `stage1_efficiency_plot.tex` : courbe d'efficacité.

## Régénération

Depuis la racine du dépôt :

```bash
python3 plots/generate_stage1_plots.py
```
