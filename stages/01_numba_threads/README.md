# Stage 01 Numba Threads

Cette étape contient la version `Numba + threads` du calcul, ainsi qu'un mode benchmark sans affichage pour mesurer le speed-up du calcul pur.

## Fichiers principaux

- `nbodies_grid_numba_parallel.py`
- `visualizer3d.py`

## Simulation interactive

Depuis la racine du dépôt :

```bash
source .venv/bin/activate
python stages/01_numba_threads/nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1
```

## Benchmark compute-only

```bash
NUMBA_NUM_THREADS=4 python stages/01_numba_threads/nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
```

## Signification des temps

- En mode interactif, `Render time` mesure la partie avant la mise à jour physique dans la boucle: gestion des événements SDL + rendu OpenGL + swap des buffers.
- En mode interactif, `Update time` mesure l'appel au calcul physique et la mise à jour des positions dans le visualiseur.
- En mode `--benchmark`, le temps mesuré correspond uniquement à `update_positions(dt)`, sans affichage et sans warmup JIT.

## Campagne de mesures recommandée

```bash
for n in 1 2 4 8 16; do
  NUMBA_NUM_THREADS=$n python stages/01_numba_threads/nbodies_grid_numba_parallel.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
done
```

## Règle

Cette étape ne doit pas modifier `stages/00_baseline/`.
