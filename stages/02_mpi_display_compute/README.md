# Stage 02 MPI Display Compute

Cette étape sépare explicitement le rendu graphique et le calcul physique avec MPI.

## Fichiers principaux

- `nbodies_grid_numba_mpi_display.py`
- `visualizer3d.py`

## Répartition des rôles

- `rank 0` : reçoit les positions, gère SDL/OpenGL et affiche la galaxie
- `rank 1` : conserve `NBodySystem`, calcule les nouveaux pas de temps et envoie les positions

Cette étape doit être exécutée exactement avec `2` processus MPI.

## Simulation interactive

Depuis la racine du dépôt :

```bash
source .venv/bin/activate
mpirun --bind-to none -np 2 python stages/02_mpi_display_compute/nbodies_grid_numba_mpi_display.py data/galaxy_5000 0.0015 15 15 1
```

## Benchmark MPI sans affichage

```bash
NUMBA_NUM_THREADS=4 mpirun --bind-to none -np 2 python stages/02_mpi_display_compute/nbodies_grid_numba_mpi_display.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
```

## Signification des temps

- En mode interactif, `Render time` mesure la partie locale du `rank 0` : événements SDL + rendu OpenGL + swap des buffers.
- En mode interactif, `MPI wait + point update time` mesure l'attente de la nouvelle trame venant du `rank 1` puis la copie des positions dans le visualiseur.
- En mode `--benchmark`, la ligne `BENCHMARK_MPI` distingue :
  - `rank1_compute_ms` : coût moyen de `update_positions(dt)` sur le `rank 1`
  - `end_to_end_ms` : coût vu par le `rank 0`, calcul distant + transfert MPI inclus
  - `mpi_overhead_ms` : différence entre les deux

## Campagne de mesures recommandée

```bash
for n in 1 2 4 8 16; do
  NUMBA_NUM_THREADS=$n mpirun --bind-to none -np 2 python stages/02_mpi_display_compute/nbodies_grid_numba_mpi_display.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
done
```

## Remarque MPI

L'option `--bind-to none` est importante sur cette machine. Sans elle, OpenMPI peut imposer une affinité processeur trop stricte et casser la montée en charge des threads Numba sur le `rank 1`.

## Règle

Cette étape ne doit pas modifier `stages/00_baseline/` ni `stages/01_numba_threads/`.
