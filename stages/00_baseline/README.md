# Stage 00 Baseline

Esta carpeta conserva la versión base ejecutable usada como referencia.

## Archivos congelados

- `nbodies_grid.py`
- `visualizer3d.py`

## Ejecución

Ejecutar desde la raíz del repositorio:

```bash
source .venv/bin/activate
python stages/00_baseline/nbodies_grid.py data/galaxy_1000 0.001 20,20,1
```

Para una carga mayor:

```bash
python stages/00_baseline/nbodies_grid.py data/galaxy_5000 0.0015 15,15,1
```

## Nota

Esta etapa no debe modificarse al avanzar a fases posteriores.
