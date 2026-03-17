# Stages Workflow

Ce répertoire conserve une copie livrable de chaque étape importante du projet.

## Règles

- Chaque étape fermée reste figée dans son propre dossier.
- Une nouvelle étape part d'une copie de l'étape précédente, sans modifier les fichiers déjà gelés.
- Les branches `stage/*` servent au développement.
- Les tags `exam-stage-*` servent à retrouver les jalons validés.

## Étapes prévues

- `00_baseline`: version de référence exécutable.
- `01_numba_threads`: parallélisation Numba avec threads.
- `02_mpi_display_compute`: séparation MPI affichage/calcul.
- `03_mpi_full_ghost`: parallélisation MPI complète avec cellules fantômes.
