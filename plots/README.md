# Plots

Ce dossier contient les scripts et les données utilisés pour générer les graphiques du rapport.

## Fichiers

- `generate_stage1_plots.py` : génère les fichiers de données CSV et les snippets LaTeX/TikZ utilisés dans le rapport.
- `generate_stage2_plots.py` : génère les données et graphiques utilisés pour l'étape 2 et pour la comparaison entre les étapes 1 et 2.
- `initial_render_update.csv` : données de la mesure interactive initiale.
- `stage1_scaling.csv` : données de benchmark de l'étape 1.
- `stage1_interactive_scaling.csv` : données de benchmark interactif de l'étape 1.
- `stage2_scaling.csv` : données de benchmark MPI de l'étape 2.
- `stage2_interactive_scaling.csv` : données interactives de l'étape 2.
- `stage1_interactive.csv` : données interactives stabilisées de l'étape 1.
- `interactive_stage1_stage2.csv` : comparaison interactive entre les étapes 1 et 2.
- `compute_speedup_stage1_stage2.csv` : comparaison des speed-up compute-only entre les étapes 1 et 2.
- `interactive_speedup_stage1_stage2.csv` : comparaison des speed-up interactifs totaux entre les étapes 1 et 2.
- `initial_render_update_plot.tex` : graphique en barres de la mesure initiale.
- `stage1_speedup_plot.tex` : courbe de speed-up.
- `stage1_efficiency_plot.tex` : courbe d'efficacité.
- `stage1_interactive_plot.tex` : décomposition des temps interactifs de l'étape 1 à 4 threads.
- `stage1_interactive_speedup_plot.tex` : courbe de speed-up interactif de l'étape 1.
- `stage1_interactive_efficiency_plot.tex` : courbe d'efficacité interactive de l'étape 1.
- `stage2_speedup_plot.tex` : courbe de speed-up bout-en-bout de l'étape 2.
- `stage2_efficiency_plot.tex` : courbe d'efficacité bout-en-bout de l'étape 2.
- `stage1_vs_stage2_plot.tex` : comparaison des temps moyens par pas entre les étapes 1 et 2.
- `compute_speedup_stage1_stage2_plot.tex` : comparaison des speed-up compute-only entre les étapes 1 et 2.
- `interactive_stage1_stage2_plot.tex` : comparaison interactive entre les étapes 1 et 2.
- `interactive_speedup_stage1_stage2_plot.tex` : comparaison des speed-up interactifs entre les étapes 1 et 2.

## Régénération

Depuis la racine du dépôt :

```bash
python3 plots/generate_stage1_plots.py
python3 plots/generate_stage2_plots.py
```
