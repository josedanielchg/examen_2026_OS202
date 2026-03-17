# Stage 03 MPI Full Ghost

Cette étape distribue réellement le calcul sur plusieurs processus MPI.

## Fichiers principaux

- `nbodies_grid_numba_mpi_full.py`
- `visualizer3d.py`

## Répartition des rôles

- `rank 0` : lit l'état initial, distribue les particules, reconstruit les positions globales et affiche la galaxie
- `ranks 1..P-1` : workers de calcul, propriétaires d'un sous-domaine 2D de la grille

Chaque particule possède un `global_id` fixe. Les workers n'échangent que les données nécessaires :
- particules propriétaires pour la migration ;
- particules fantômes pour les interactions proches ;
- masses et centres de masse globaux par `Allreduce` pour les cellules lointaines.

## Contraintes MPI

- cette étape nécessite au moins `5` processus MPI au total ;
- les workers doivent former une vraie grille 2D `px x py`, avec `px > 1` et `py > 1` ;
- exemples valides sur cette machine :
  - `-np 5` : `4` workers = `2 x 2`
  - `-np 7` : `6` workers = `3 x 2`
  - `-np 9` : `8` workers = `4 x 2`

Le cas à `2` processus reste celui de `stages/02_mpi_display_compute/`.

## Simulation interactive

Depuis la racine du dépôt :

```bash
source .venv/bin/activate
mpirun --bind-to none -np 5 python stages/03_mpi_full_ghost/nbodies_grid_numba_mpi_full.py data/galaxy_1000 0.0015 15 15 1
```

Pour `-np 9`, ajouter aussi `--use-hwthread-cpus` :

```bash
mpirun --use-hwthread-cpus --bind-to none -np 9 python stages/03_mpi_full_ghost/nbodies_grid_numba_mpi_full.py data/galaxy_1000 0.0015 15 15 1
```

## Benchmark distribué sans affichage

```bash
NUMBA_NUM_THREADS=2 mpirun --bind-to none -np 5 python stages/03_mpi_full_ghost/nbodies_grid_numba_mpi_full.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
```

Pour `8` workers :

```bash
NUMBA_NUM_THREADS=1 mpirun --use-hwthread-cpus --bind-to none -np 9 python stages/03_mpi_full_ghost/nbodies_grid_numba_mpi_full.py data/galaxy_5000 0.0015 15 15 1 --benchmark --steps 30 --warmup 1
```

## Signification des temps

- en mode interactif, `Render time` mesure la partie locale du `rank 0` : événements SDL + rendu OpenGL + swap ;
- en mode interactif, `Distributed step + point update time` mesure l'attente du nouveau pas distribué puis la copie des positions dans le visualiseur ;
- en mode `--benchmark`, la ligne `BENCHMARK_MPI_FULL` distingue :
  - `end_to_end_ms` : coût total vu par le `rank 0` ;
  - `gather_ms` : coût de reconstruction des positions globales ;
  - `worker_step_mean_ms` : temps moyen d'un worker ;
  - `worker_step_max_ms` : temps du worker critique ;
  - `migration_ms`, `halo_ms`, `reduce_ms` : sous-coûts MPI principaux ;
  - `imbalance_ms` : différence entre worker critique et worker moyen.

## Remarques pratiques

- l'option `--bind-to none` reste importante pour ne pas limiter les threads Numba ;
- pour `-np 9`, OpenMPI demande `--use-hwthread-cpus` sur cette machine ;
- la configuration `8 workers / 2 threads` a été observée comme instable pendant la campagne officielle et n'est donc pas retenue dans le rapport.

## Règle

Cette étape ne doit pas modifier `stages/00_baseline/`, `stages/01_numba_threads/` ni `stages/02_mpi_display_compute/`.
