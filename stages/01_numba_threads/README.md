# Stage 01 Numba Threads

Cette étape parallélise le calcul en mémoire partagée avec `Numba`.

## Fichiers principaux

- `nbodies_grid_numba_parallel.py`
- `visualizer3d.py`

## Simulation interactive

Depuis la racine du dépôt :

```bash
source .venv/bin/activate
python stages/01_numba_threads/nbodies_grid_numba_parallel.py data/galaxy_1000 0.0015 15 15 1
```

## Benchmark sans affichage

```bash
NUMBA_NUM_THREADS=4 python stages/01_numba_threads/nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
```

## Signification des temps

- en mode interactif, `Render time` mesure la partie locale du visualiseur : événements SDL + rendu OpenGL + swap ;
- en mode interactif, `Update time` mesure le calcul physique puis la mise à jour des positions dans le visualiseur ;
- en mode `--benchmark`, le temps mesuré correspond seulement à `update_positions(dt)`, sans SDL/OpenGL.

## Campagne recommandée

```bash
for n in 1 2 4 8 16; do
  NUMBA_NUM_THREADS=$n python stages/01_numba_threads/nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
done
```

## Règle

Cette étape est une copie de travail indépendante et ne doit pas modifier `stages/00_baseline/`.
