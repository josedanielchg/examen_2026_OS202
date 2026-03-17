# Entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Flujo por etapas

La entrega final debe conservar cada implementación en `stages/`.

## Etapa 00: baseline congelada

```bash
python stages/00_baseline/nbodies_grid.py data/galaxy_1000 0.001 20,20,1
python stages/00_baseline/nbodies_grid.py data/galaxy_5000 0.0015 15,15,1
```

## Etapa 01: numba con threads

Pendiente de implementación en `stages/01_numba_threads/`.

## Etapa 02: MPI display/cálculo

Pendiente de implementación en `stages/02_mpi_display_compute/`.

## Etapa 03: MPI completo con ghost cells

Pendiente de implementación en `stages/03_mpi_full_ghost/`.

# Scripts auxiliares del repositorio

## Generar datos

```bash
python galaxy_generator.py 10000 data/galaxy_10000
```

## Simulacion con grilla

```bash
python nbodies_grid.py data/galaxy_1000 0.001 20,20,1
```

## Simulacion con grilla y numba

```bash
python nbodies_grid_numba.py data/galaxy_5000 0.0015 15 15 1
```

## Simulacion Barnes-Hut

```bash
python barnes_hut_numba.py data/galaxy_1000 0.001 0.5
```

## Visualizadores de prueba

```bash
python visualizer3d.py
python visualizer3d_vbo.py
python visualizer3d_sans_vbo.py
```

# Opcional: fijar hilos de numba

```bash
NUMBA_NUM_THREADS=4 python nbodies_grid_numba.py data/galaxy_5000 0.0015 15 15 1
NUMBA_NUM_THREADS=4 python barnes_hut_numba.py data/galaxy_1000 0.001 0.5
```

# Nota

En esta maquina los imports de `numpy`, `numba`, `PySDL2` y `PyOpenGL` ya quedaron verificados dentro de `.venv`.

Si ejecutas esto en otra maquina Linux y falta SDL2 a nivel sistema, instala el paquete de tu distribucion correspondiente a SDL2 antes de abrir los visualizadores.

Las ramas `stage/*` se usan para desarrollo temporal. La preservación real de cada implementación debe quedar dentro de `stages/`.
