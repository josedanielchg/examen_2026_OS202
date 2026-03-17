# Simulation d'une galaxie à n corps avec répartition MPI 2D des cellules de grille.
# Stage 03 :
# - rank 0 : affichage et coordination
# - ranks 1..P-1 : calcul distribué, migration des particules et cellules fantômes
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import visualizer3d
from mpi4py import MPI
from numba import get_num_threads, njit, prange


G = 1.560339e-13
HALO_WIDTH = 2

TAG_INIT = 50
TAG_MIG_IDS = 60
TAG_MIG_POS = 61
TAG_MIG_VEL = 62
TAG_MIG_MASS = 63
TAG_HALO_COUNT = 70
TAG_HALO_IDS = 71
TAG_HALO_POS = 72
TAG_HALO_MASS = 73

CMD_STEP = 1
CMD_STOP = 2

MPI_INT64 = MPI.INT64_T if hasattr(MPI, "INT64_T") else MPI.LONG_LONG
DEBUG_STAGE3 = os.environ.get("STAGE3_DEBUG", "0") == "1"


def debug_log(rank: int, message: str):
    if DEBUG_STAGE3:
        print(f"[stage03][rank {rank}] {message}", flush=True)


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


@njit(cache=True, parallel=True)
def integrate_positions(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, dt: float):
    for ibody in prange(positions.shape[0]):
        positions[ibody, 0] += velocities[ibody, 0] * dt + 0.5 * accelerations[ibody, 0] * dt * dt
        positions[ibody, 1] += velocities[ibody, 1] * dt + 0.5 * accelerations[ibody, 1] * dt * dt
        positions[ibody, 2] += velocities[ibody, 2] * dt + 0.5 * accelerations[ibody, 2] * dt * dt


@njit(cache=True, parallel=True)
def integrate_velocities(velocities: np.ndarray, accelerations: np.ndarray, accelerations_new: np.ndarray, dt: float):
    for ibody in prange(velocities.shape[0]):
        velocities[ibody, 0] += 0.5 * (accelerations[ibody, 0] + accelerations_new[ibody, 0]) * dt
        velocities[ibody, 1] += 0.5 * (accelerations[ibody, 1] + accelerations_new[ibody, 1]) * dt
        velocities[ibody, 2] += 0.5 * (accelerations[ibody, 2] + accelerations_new[ibody, 2]) * dt


@njit(cache=True, parallel=True)
def compute_acceleration_distributed(
    owned_positions: np.ndarray,
    owned_ids: np.ndarray,
    available_positions: np.ndarray,
    available_masses: np.ndarray,
    available_ids: np.ndarray,
    cell_start_indices: np.ndarray,
    sorted_particle_indices: np.ndarray,
    cell_masses: np.ndarray,
    cell_com_positions: np.ndarray,
    grid_min: np.ndarray,
    cell_size: np.ndarray,
    n_cells: np.ndarray,
):
    accelerations = np.zeros_like(owned_positions)
    for ibody in prange(owned_positions.shape[0]):
        pos = owned_positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for axis in range(3):
            if cell_idx[axis] < 0:
                cell_idx[axis] = 0
            elif cell_idx[axis] >= n_cells[axis]:
                cell_idx[axis] = n_cells[axis] - 1

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    if (abs(ix - cell_idx[0]) > HALO_WIDTH) or (abs(iy - cell_idx[1]) > HALO_WIDTH) or (abs(iz - cell_idx[2]) > HALO_WIDTH):
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com_positions[morse_idx] - pos
                            distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                accelerations[ibody, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            particle_idx = sorted_particle_indices[j]
                            if available_ids[particle_idx] != owned_ids[ibody]:
                                direction = available_positions[particle_idx] - pos
                                distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    accelerations[ibody, :] += G * direction[:] * inv_dist3 * available_masses[particle_idx]
    return accelerations


@dataclass
class InitialState:
    global_ids: np.ndarray
    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    colors: np.ndarray
    luminosities: np.ndarray
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    grid_min: np.ndarray
    grid_max: np.ndarray
    cell_size: np.ndarray
    n_cells: np.ndarray


def load_initial_state(filename: str, n_cells_per_dir: tuple[int, int, int]) -> InitialState:
    positions = []
    velocities = []
    masses = []

    box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)
    with open(filename, "r", encoding="utf-8") as fich:
        for line in fich:
            data = line.split()
            if not data:
                continue
            masses.append(float(data[0]))
            positions.append([float(data[1]), float(data[2]), float(data[3])])
            velocities.append([float(data[4]), float(data[5]), float(data[6])])
            for axis in range(3):
                box[0][axis] = min(box[0][axis], positions[-1][axis] - 1.0e-6)
                box[1][axis] = max(box[1][axis], positions[-1][axis] + 1.0e-6)

    positions_array = np.array(positions, dtype=np.float64)
    velocities_array = np.array(velocities, dtype=np.float64)
    masses_array = np.array(masses, dtype=np.float64)
    max_mass = float(np.max(masses_array))
    colors = np.array([generate_star_color(mass) for mass in masses], dtype=np.float32)
    luminosities = np.clip(masses_array / max_mass, 0.5, 1.0).astype(np.float32)
    grid_min = np.min(positions_array, axis=0) - 1.0e-6
    grid_max = np.max(positions_array, axis=0) + 1.0e-6
    n_cells = np.array(n_cells_per_dir, dtype=np.int64)
    cell_size = (grid_max - grid_min) / n_cells
    bounds = (
        (float(box[0][0]), float(box[1][0])),
        (float(box[0][1]), float(box[1][1])),
        (float(box[0][2]), float(box[1][2])),
    )
    return InitialState(
        global_ids=np.arange(masses_array.shape[0], dtype=np.int64),
        masses=masses_array,
        positions=positions_array,
        velocities=velocities_array,
        colors=colors,
        luminosities=luminosities,
        bounds=bounds,
        grid_min=grid_min.astype(np.float64),
        grid_max=grid_max.astype(np.float64),
        cell_size=cell_size.astype(np.float64),
        n_cells=n_cells,
    )


def choose_worker_grid(nworkers: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    best_gap: int | None = None
    for py in range(2, int(math.sqrt(nworkers)) + 1):
        if nworkers % py != 0:
            continue
        px = nworkers // py
        if px <= 1:
            continue
        gap = abs(px - py)
        if best is None or gap < best_gap:
            best = (px, py)
            best_gap = gap
    return best


def split_axis(ncells: int, nparts: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.empty(nparts, dtype=np.int64)
    ends = np.empty(nparts, dtype=np.int64)
    base = ncells // nparts
    remainder = ncells % nparts
    current = 0
    for part in range(nparts):
        width = base + (1 if part < remainder else 0)
        starts[part] = current
        current += width
        ends[part] = current
    return starts, ends


def compute_cell_indices(positions: np.ndarray, grid_min: np.ndarray, cell_size: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    if positions.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)
    cell_indices = np.floor((positions - grid_min) / cell_size).astype(np.int64)
    np.clip(cell_indices, 0, n_cells - 1, out=cell_indices)
    return cell_indices


def compute_morse_ids(cell_indices: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    if cell_indices.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    return cell_indices[:, 0] + cell_indices[:, 1] * n_cells[0] + cell_indices[:, 2] * n_cells[0] * n_cells[1]


def owner_compute_ranks_from_cells(cell_indices: np.ndarray, x_ends: np.ndarray, y_ends: np.ndarray, px: int) -> np.ndarray:
    if cell_indices.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    x_parts = np.searchsorted(x_ends, cell_indices[:, 0], side="right")
    y_parts = np.searchsorted(y_ends, cell_indices[:, 1], side="right")
    return (y_parts * px + x_parts).astype(np.int64)


def gather_positions_root(comm, total_bodies: int) -> np.ndarray:
    debug_log(0, "collecting counts for gather")
    counts = np.array(comm.gather(0, root=0), dtype=np.int32)
    displs = np.zeros_like(counts)
    if counts.shape[0] > 1:
        displs[1:] = np.cumsum(counts[:-1])

    total_owned = int(np.sum(counts))
    gathered_ids = np.empty((total_owned,), dtype=np.int64)
    gathered_positions = np.empty((total_owned * 3,), dtype=np.float64)
    empty_ids = np.empty((0,), dtype=np.int64)
    empty_positions = np.empty((0,), dtype=np.float64)

    debug_log(0, f"gatherv ids with counts={counts.tolist()}")
    comm.Gatherv([empty_ids, MPI_INT64], [gathered_ids, counts, displs, MPI_INT64], root=0)

    pos_counts = counts * 3
    pos_displs = np.zeros_like(pos_counts)
    if pos_counts.shape[0] > 1:
        pos_displs[1:] = np.cumsum(pos_counts[:-1])
    debug_log(0, "gatherv positions")
    comm.Gatherv([empty_positions, MPI.DOUBLE], [gathered_positions, pos_counts, pos_displs, MPI.DOUBLE], root=0)

    if total_owned != total_bodies:
        raise RuntimeError(f"Nombre de particules incohérent sur le root : {total_owned} au lieu de {total_bodies}.")

    sorted_ids = np.sort(gathered_ids)
    if not np.array_equal(sorted_ids, np.arange(total_bodies, dtype=np.int64)):
        raise RuntimeError("Les global_id propriétaires ne couvrent pas exactement l'ensemble des particules.")

    ordered_positions = np.empty((total_bodies, 3), dtype=np.float64)
    ordered_positions[gathered_ids] = gathered_positions.reshape((-1, 3))
    if not np.isfinite(ordered_positions).all():
        raise RuntimeError("Valeur non finie détectée dans les positions reconstruites sur le rank 0.")
    return ordered_positions


def gather_positions_worker(comm, local_ids: np.ndarray, local_positions: np.ndarray):
    debug_log(comm.Get_rank(), f"sending gather count={int(local_ids.shape[0])}")
    comm.gather(int(local_ids.shape[0]), root=0)
    debug_log(comm.Get_rank(), "enter gatherv ids")
    comm.Gatherv([local_ids, MPI_INT64], None, root=0)
    debug_log(comm.Get_rank(), "enter gatherv positions")
    comm.Gatherv([local_positions.reshape(-1), MPI.DOUBLE], None, root=0)


class DistributedGridWorker:
    def __init__(self, compute_comm, init_payload: dict):
        self.compute_comm = compute_comm
        self.compute_rank = init_payload["compute_rank"]
        self.nworkers = init_payload["nworkers"]
        self.total_bodies = init_payload["total_bodies"]
        self.grid_min = np.array(init_payload["grid_min"], dtype=np.float64)
        self.grid_max = np.array(init_payload["grid_max"], dtype=np.float64)
        self.cell_size = np.array(init_payload["cell_size"], dtype=np.float64)
        self.n_cells = np.array(init_payload["n_cells"], dtype=np.int64)
        self.total_cells = int(np.prod(self.n_cells))
        self.px = init_payload["px"]
        self.py = init_payload["py"]
        self.x_starts = np.array(init_payload["x_starts"], dtype=np.int64)
        self.x_ends = np.array(init_payload["x_ends"], dtype=np.int64)
        self.y_starts = np.array(init_payload["y_starts"], dtype=np.int64)
        self.y_ends = np.array(init_payload["y_ends"], dtype=np.int64)

        self.coord_x = self.compute_rank % self.px
        self.coord_y = self.compute_rank // self.px
        self.x_start = int(self.x_starts[self.coord_x])
        self.x_end = int(self.x_ends[self.coord_x])
        self.y_start = int(self.y_starts[self.coord_y])
        self.y_end = int(self.y_ends[self.coord_y])

        self.neighbors = self._build_neighbors()

        self.owned_global_ids = np.array(init_payload["global_ids"], dtype=np.int64)
        self.owned_positions = np.array(init_payload["positions"], dtype=np.float64)
        self.owned_velocities = np.array(init_payload["velocities"], dtype=np.float64)
        self.owned_masses = np.array(init_payload["masses"], dtype=np.float64)

        self.ghost_global_ids = np.empty((0,), dtype=np.int64)
        self.ghost_positions = np.empty((0, 3), dtype=np.float64)
        self.ghost_masses = np.empty((0,), dtype=np.float64)

        self.owned_cell_indices = np.empty((0, 3), dtype=np.int64)
        self.owned_cell_morse = np.empty((0,), dtype=np.int64)
        self.available_global_ids = np.empty((0,), dtype=np.int64)
        self.available_positions = np.empty((0, 3), dtype=np.float64)
        self.available_masses = np.empty((0,), dtype=np.float64)
        self.cell_start_indices = np.zeros((self.total_cells + 1,), dtype=np.int64)
        self.sorted_particle_indices = np.empty((0,), dtype=np.int64)
        self.cell_masses = np.zeros((self.total_cells,), dtype=np.float64)
        self.cell_com_positions = np.zeros((self.total_cells, 3), dtype=np.float64)

        self._refresh_static_state()

    def _build_neighbors(self) -> list[tuple[int, int, int]]:
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = self.coord_x + dx
                ny = self.coord_y + dy
                if 0 <= nx < self.px and 0 <= ny < self.py:
                    neighbors.append((ny * self.px + nx, dx, dy))
        return neighbors

    def _refresh_owned_cells(self):
        self.owned_cell_indices = compute_cell_indices(self.owned_positions, self.grid_min, self.cell_size, self.n_cells)
        self.owned_cell_morse = compute_morse_ids(self.owned_cell_indices, self.n_cells)

    def _refresh_static_state(self):
        debug_log(self.compute_rank + 1, "refresh static state: owned cells")
        self._refresh_owned_cells()
        debug_log(self.compute_rank + 1, "refresh static state: exchange ghosts")
        self.exchange_ghost_particles()
        debug_log(self.compute_rank + 1, "refresh static state: reduce cells")
        self.reduce_global_cells()
        debug_log(self.compute_rank + 1, "refresh static state: rebuild available")
        self.rebuild_available_particles()

    def migrate_owned_particles(self) -> float:
        t0 = time.perf_counter()
        destination_ranks = owner_compute_ranks_from_cells(self.owned_cell_indices, self.x_ends, self.y_ends, self.px)
        send_counts = np.bincount(destination_ranks, minlength=self.nworkers).astype(np.int32)
        recv_counts = self.compute_comm.alltoall(send_counts.tolist())

        keep_mask = destination_ranks == self.compute_rank
        kept_ids = self.owned_global_ids[keep_mask]
        kept_positions = self.owned_positions[keep_mask]
        kept_velocities = self.owned_velocities[keep_mask]
        kept_masses = self.owned_masses[keep_mask]

        recv_buffers = []
        recv_requests = []
        for source_rank, recv_count in enumerate(recv_counts):
            if source_rank == self.compute_rank or recv_count == 0:
                continue
            recv_ids = np.empty((recv_count,), dtype=np.int64)
            recv_positions = np.empty((recv_count, 3), dtype=np.float64)
            recv_velocities = np.empty((recv_count, 3), dtype=np.float64)
            recv_masses = np.empty((recv_count,), dtype=np.float64)
            recv_requests.extend(
                [
                    self.compute_comm.Irecv([recv_ids, MPI_INT64], source=source_rank, tag=TAG_MIG_IDS),
                    self.compute_comm.Irecv([recv_positions, MPI.DOUBLE], source=source_rank, tag=TAG_MIG_POS),
                    self.compute_comm.Irecv([recv_velocities, MPI.DOUBLE], source=source_rank, tag=TAG_MIG_VEL),
                    self.compute_comm.Irecv([recv_masses, MPI.DOUBLE], source=source_rank, tag=TAG_MIG_MASS),
                ]
            )
            recv_buffers.append((recv_ids, recv_positions, recv_velocities, recv_masses))

        for dest_rank in range(self.nworkers):
            if dest_rank == self.compute_rank or send_counts[dest_rank] == 0:
                continue
            send_mask = destination_ranks == dest_rank
            self.compute_comm.Send([self.owned_global_ids[send_mask], MPI_INT64], dest=dest_rank, tag=TAG_MIG_IDS)
            self.compute_comm.Send([self.owned_positions[send_mask], MPI.DOUBLE], dest=dest_rank, tag=TAG_MIG_POS)
            self.compute_comm.Send([self.owned_velocities[send_mask], MPI.DOUBLE], dest=dest_rank, tag=TAG_MIG_VEL)
            self.compute_comm.Send([self.owned_masses[send_mask], MPI.DOUBLE], dest=dest_rank, tag=TAG_MIG_MASS)

        if recv_requests:
            MPI.Request.Waitall(recv_requests)

        received_ids = [buffer[0] for buffer in recv_buffers]
        received_positions = [buffer[1] for buffer in recv_buffers]
        received_velocities = [buffer[2] for buffer in recv_buffers]
        received_masses = [buffer[3] for buffer in recv_buffers]

        if received_ids:
            self.owned_global_ids = np.concatenate([kept_ids] + received_ids)
            self.owned_positions = np.concatenate([kept_positions] + received_positions)
            self.owned_velocities = np.concatenate([kept_velocities] + received_velocities)
            self.owned_masses = np.concatenate([kept_masses] + received_masses)
        else:
            self.owned_global_ids = kept_ids
            self.owned_positions = kept_positions
            self.owned_velocities = kept_velocities
            self.owned_masses = kept_masses

        self._refresh_owned_cells()
        return time.perf_counter() - t0

    def exchange_ghost_particles(self) -> float:
        t0 = time.perf_counter()
        send_payloads: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for neighbor_rank, dx, dy in self.neighbors:
            mask = np.ones((self.owned_global_ids.shape[0],), dtype=bool)
            if dx < 0:
                mask &= self.owned_cell_indices[:, 0] < self.x_start + HALO_WIDTH
            elif dx > 0:
                mask &= self.owned_cell_indices[:, 0] >= self.x_end - HALO_WIDTH
            if dy < 0:
                mask &= self.owned_cell_indices[:, 1] < self.y_start + HALO_WIDTH
            elif dy > 0:
                mask &= self.owned_cell_indices[:, 1] >= self.y_end - HALO_WIDTH
            send_payloads[neighbor_rank] = (
                self.owned_global_ids[mask].copy(),
                self.owned_positions[mask].copy(),
                self.owned_masses[mask].copy(),
            )

        recv_count_buffers = {}
        recv_count_requests = []
        for neighbor_rank, _dx, _dy in self.neighbors:
            buffer = np.zeros((1,), dtype=np.int32)
            recv_count_buffers[neighbor_rank] = buffer
            recv_count_requests.append(self.compute_comm.Irecv([buffer, MPI.INT], source=neighbor_rank, tag=TAG_HALO_COUNT))

        for neighbor_rank, payload in send_payloads.items():
            count = np.array([payload[0].shape[0]], dtype=np.int32)
            self.compute_comm.Send([count, MPI.INT], dest=neighbor_rank, tag=TAG_HALO_COUNT)

        if recv_count_requests:
            MPI.Request.Waitall(recv_count_requests)

        recv_buffers = []
        recv_requests = []
        for neighbor_rank, buffer in recv_count_buffers.items():
            recv_count = int(buffer[0])
            if recv_count == 0:
                continue
            recv_ids = np.empty((recv_count,), dtype=np.int64)
            recv_positions = np.empty((recv_count, 3), dtype=np.float64)
            recv_masses = np.empty((recv_count,), dtype=np.float64)
            recv_requests.extend(
                [
                    self.compute_comm.Irecv([recv_ids, MPI_INT64], source=neighbor_rank, tag=TAG_HALO_IDS),
                    self.compute_comm.Irecv([recv_positions, MPI.DOUBLE], source=neighbor_rank, tag=TAG_HALO_POS),
                    self.compute_comm.Irecv([recv_masses, MPI.DOUBLE], source=neighbor_rank, tag=TAG_HALO_MASS),
                ]
            )
            recv_buffers.append((recv_ids, recv_positions, recv_masses))

        for neighbor_rank, payload in send_payloads.items():
            ids, positions, masses = payload
            if ids.shape[0] == 0:
                continue
            self.compute_comm.Send([ids, MPI_INT64], dest=neighbor_rank, tag=TAG_HALO_IDS)
            self.compute_comm.Send([positions, MPI.DOUBLE], dest=neighbor_rank, tag=TAG_HALO_POS)
            self.compute_comm.Send([masses, MPI.DOUBLE], dest=neighbor_rank, tag=TAG_HALO_MASS)

        if recv_requests:
            MPI.Request.Waitall(recv_requests)

        if recv_buffers:
            self.ghost_global_ids = np.concatenate([buffer[0] for buffer in recv_buffers])
            self.ghost_positions = np.concatenate([buffer[1] for buffer in recv_buffers])
            self.ghost_masses = np.concatenate([buffer[2] for buffer in recv_buffers])
        else:
            self.ghost_global_ids = np.empty((0,), dtype=np.int64)
            self.ghost_positions = np.empty((0, 3), dtype=np.float64)
            self.ghost_masses = np.empty((0,), dtype=np.float64)
        return time.perf_counter() - t0

    def reduce_global_cells(self) -> float:
        t0 = time.perf_counter()
        local_masses = np.zeros((self.total_cells,), dtype=np.float64)
        local_weighted_positions = np.zeros((self.total_cells, 3), dtype=np.float64)
        if self.owned_global_ids.shape[0] > 0:
            owned_masses64 = self.owned_masses.astype(np.float64)
            np.add.at(local_masses, self.owned_cell_morse, owned_masses64)
            np.add.at(local_weighted_positions, self.owned_cell_morse, self.owned_positions.astype(np.float64) * owned_masses64[:, np.newaxis])

        global_masses = np.empty_like(local_masses)
        global_weighted_positions = np.empty_like(local_weighted_positions)
        self.compute_comm.Allreduce([local_masses, MPI.DOUBLE], [global_masses, MPI.DOUBLE], op=MPI.SUM)
        self.compute_comm.Allreduce([local_weighted_positions, MPI.DOUBLE], [global_weighted_positions, MPI.DOUBLE], op=MPI.SUM)

        self.cell_masses = global_masses
        self.cell_com_positions.fill(0.0)
        non_empty = global_masses > 0.0
        self.cell_com_positions[non_empty] = global_weighted_positions[non_empty] / global_masses[non_empty, np.newaxis]
        return time.perf_counter() - t0

    def rebuild_available_particles(self):
        if self.ghost_global_ids.shape[0] > 0:
            self.available_global_ids = np.concatenate([self.owned_global_ids, self.ghost_global_ids])
            self.available_positions = np.concatenate([self.owned_positions, self.ghost_positions])
            self.available_masses = np.concatenate([self.owned_masses, self.ghost_masses])
        else:
            self.available_global_ids = np.array(self.owned_global_ids, copy=True)
            self.available_positions = np.array(self.owned_positions, copy=True)
            self.available_masses = np.array(self.owned_masses, copy=True)

        available_cell_indices = compute_cell_indices(self.available_positions, self.grid_min, self.cell_size, self.n_cells)
        available_cell_morse = compute_morse_ids(available_cell_indices, self.n_cells)
        counts = np.bincount(available_cell_morse, minlength=self.total_cells) if available_cell_morse.shape[0] > 0 else np.zeros((self.total_cells,), dtype=np.int64)
        self.cell_start_indices = np.empty((self.total_cells + 1,), dtype=np.int64)
        self.cell_start_indices[0] = 0
        self.cell_start_indices[1:] = np.cumsum(counts, dtype=np.int64)
        self.sorted_particle_indices = np.argsort(available_cell_morse, kind="mergesort").astype(np.int64) if available_cell_morse.shape[0] > 0 else np.empty((0,), dtype=np.int64)

    def compute_acceleration(self) -> np.ndarray:
        if self.owned_positions.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        return compute_acceleration_distributed(
            self.owned_positions,
            self.owned_global_ids,
            self.available_positions,
            self.available_masses,
            self.available_global_ids,
            self.cell_start_indices,
            self.sorted_particle_indices,
            self.cell_masses,
            self.cell_com_positions,
            self.grid_min,
            self.cell_size,
            self.n_cells,
        )

    def step(self, dt: float) -> dict:
        step_start = time.perf_counter()
        a_old = self.compute_acceleration()
        integrate_positions(self.owned_positions, self.owned_velocities, a_old, dt)
        if not np.isfinite(self.owned_positions).all():
            raise RuntimeError(f"Positions non finies détectées sur le worker {self.compute_rank}.")
        migration_s = self.migrate_owned_particles()
        halo_s = self.exchange_ghost_particles()
        reduce_s = self.reduce_global_cells()
        self.rebuild_available_particles()
        a_new = self.compute_acceleration()
        integrate_velocities(self.owned_velocities, a_old, a_new, dt)
        if not np.isfinite(self.owned_velocities).all():
            raise RuntimeError(f"Vitesses non finies détectées sur le worker {self.compute_rank}.")
        step_s = time.perf_counter() - step_start
        return {
            "step_s": float(step_s),
            "migration_s": float(migration_s),
            "halo_s": float(halo_s),
            "reduce_s": float(reduce_s),
        }


def build_initial_payloads(state: InitialState, px: int, py: int) -> tuple[dict, list[dict]]:
    x_starts, x_ends = split_axis(int(state.n_cells[0]), px)
    y_starts, y_ends = split_axis(int(state.n_cells[1]), py)
    cell_indices = compute_cell_indices(state.positions, state.grid_min, state.cell_size, state.n_cells)
    owners = owner_compute_ranks_from_cells(cell_indices, x_ends, y_ends, px)
    common_meta = {
        "grid_min": state.grid_min,
        "grid_max": state.grid_max,
        "cell_size": state.cell_size,
        "n_cells": state.n_cells,
        "px": px,
        "py": py,
        "nworkers": px * py,
        "x_starts": x_starts,
        "x_ends": x_ends,
        "y_starts": y_starts,
        "y_ends": y_ends,
        "total_bodies": state.global_ids.shape[0],
    }
    worker_payloads = []
    for compute_rank in range(px * py):
        mask = owners == compute_rank
        payload = dict(common_meta)
        payload.update(
            {
                "compute_rank": compute_rank,
                "global_ids": state.global_ids[mask],
                "positions": state.positions[mask],
                "velocities": state.velocities[mask],
                "masses": state.masses[mask],
            }
        )
        worker_payloads.append(payload)
    return common_meta, worker_payloads


def send_initial_payloads(comm, worker_payloads: list[dict]):
    for compute_rank, payload in enumerate(worker_payloads):
        comm.send(payload, dest=compute_rank + 1, tag=TAG_INIT)


def receive_initial_payload(comm) -> dict:
    debug_log(comm.Get_rank(), "waiting init payload")
    return comm.recv(source=0, tag=TAG_INIT)


def run_display_rank_interactive(comm, state: InitialState, dt: float):
    visualizer = visualizer3d.Visualizer3D(state.positions, state.colors, state.luminosities, state.bounds)

    def updater(_dt: float) -> np.ndarray:
        comm.bcast(CMD_STEP, root=0)
        return gather_positions_root(comm, state.global_ids.shape[0])

    try:
        visualizer.run(updater=updater, dt=dt)
    finally:
        comm.bcast(CMD_STOP, root=0)


def run_worker_rank_interactive(world_comm, compute_comm, init_payload: dict, dt: float):
    debug_log(world_comm.Get_rank(), "init worker interactive")
    worker = DistributedGridWorker(compute_comm, init_payload)
    while True:
        command = world_comm.bcast(None, root=0)
        if command == CMD_STOP:
            break
        if command != CMD_STEP:
            raise RuntimeError(f"Commande inconnue reçue par un worker: {command}")
        worker.step(dt)
        gather_positions_worker(world_comm, worker.owned_global_ids, worker.owned_positions)


def run_display_rank_benchmark(comm, state: InitialState, steps: int, warmup: int):
    end_to_end_times = np.empty((steps,), dtype=np.float64)
    gather_times = np.empty((steps,), dtype=np.float64)
    total_steps = warmup + steps
    for istep in range(total_steps):
        debug_log(0, f"benchmark step {istep}: broadcast step")
        t0 = time.perf_counter()
        comm.bcast(CMD_STEP, root=0)
        t_gather_0 = time.perf_counter()
        debug_log(0, f"benchmark step {istep}: gather root")
        gather_positions_root(comm, state.global_ids.shape[0])
        gather_elapsed = time.perf_counter() - t_gather_0
        elapsed = time.perf_counter() - t0
        if istep >= warmup:
            end_to_end_times[istep - warmup] = elapsed
            gather_times[istep - warmup] = gather_elapsed

    comm.bcast(CMD_STOP, root=0)
    worker_stats = comm.gather(None, root=0)
    per_worker_stats = [stat for stat in worker_stats[1:] if stat is not None]
    if not per_worker_stats:
        raise RuntimeError("Aucune statistique worker reçue en benchmark Stage 03.")

    mean_worker_step = float(np.mean([stat["mean_step_s"] for stat in per_worker_stats]))
    max_worker_step = float(np.max([stat["mean_step_s"] for stat in per_worker_stats]))
    mean_migration = float(np.mean([stat["mean_migration_s"] for stat in per_worker_stats]))
    mean_halo = float(np.mean([stat["mean_halo_s"] for stat in per_worker_stats]))
    mean_reduce = float(np.mean([stat["mean_reduce_s"] for stat in per_worker_stats]))
    threads = per_worker_stats[0]["threads"]

    print(
        "BENCHMARK_MPI_FULL "
        f"workers={len(per_worker_stats)} "
        f"threads={threads} "
        f"warmup={warmup} "
        f"steps={steps} "
        f"end_to_end_ms={np.mean(end_to_end_times) * 1000.0:.3f} "
        f"gather_ms={np.mean(gather_times) * 1000.0:.3f} "
        f"worker_step_mean_ms={mean_worker_step * 1000.0:.3f} "
        f"worker_step_max_ms={max_worker_step * 1000.0:.3f} "
        f"migration_ms={mean_migration * 1000.0:.3f} "
        f"halo_ms={mean_halo * 1000.0:.3f} "
        f"reduce_ms={mean_reduce * 1000.0:.3f} "
        f"imbalance_ms={(max_worker_step - mean_worker_step) * 1000.0:.3f}"
    )


def run_worker_rank_benchmark(world_comm, compute_comm, init_payload: dict, dt: float, steps: int, warmup: int):
    debug_log(world_comm.Get_rank(), "init worker benchmark")
    worker = DistributedGridWorker(compute_comm, init_payload)
    measured_step_times = []
    measured_migration_times = []
    measured_halo_times = []
    measured_reduce_times = []
    total_steps = warmup + steps
    current_step = 0
    while True:
        debug_log(world_comm.Get_rank(), f"waiting command at step {current_step}")
        command = world_comm.bcast(None, root=0)
        if command == CMD_STOP:
            break
        if command != CMD_STEP:
            raise RuntimeError(f"Commande inconnue reçue par un worker: {command}")
        debug_log(world_comm.Get_rank(), f"start worker step {current_step}")
        stats = worker.step(dt)
        if current_step >= warmup:
            measured_step_times.append(stats["step_s"])
            measured_migration_times.append(stats["migration_s"])
            measured_halo_times.append(stats["halo_s"])
            measured_reduce_times.append(stats["reduce_s"])
        current_step += 1
        debug_log(world_comm.Get_rank(), f"finished worker step {current_step}, start gather")
        gather_positions_worker(world_comm, worker.owned_global_ids, worker.owned_positions)
        if current_step > total_steps:
            raise RuntimeError("Nombre de pas benchmark incohérent côté worker.")

    world_comm.gather(
        {
            "threads": get_num_threads(),
            "mean_step_s": float(np.mean(measured_step_times)),
            "mean_migration_s": float(np.mean(measured_migration_times)),
            "mean_halo_s": float(np.mean(measured_halo_times)),
            "mean_reduce_s": float(np.mean(measured_reduce_times)),
        },
        root=0,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulation N-corps Stage 03 : calcul MPI distribué sur une grille 2D avec cellules fantômes."
    )
    parser.add_argument("filename", nargs="?", default="data/galaxy_1000")
    parser.add_argument("dt", nargs="?", type=float, default=0.001)
    parser.add_argument("ni", nargs="?", type=int, default=15)
    parser.add_argument("nj", nargs="?", type=int, default=15)
    parser.add_argument("nk", nargs="?", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true", help="Exécute le benchmark distribué sans ouvrir la fenêtre.")
    parser.add_argument("--steps", type=int, default=30, help="Nombre de pas mesurés en mode benchmark.")
    parser.add_argument("--warmup", type=int, default=1, help="Nombre de pas non mesurés pour amortir JIT.")
    return parser.parse_args()


def main():
    args = parse_args()
    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()
    nworkers = world_size - 1
    compute_comm = world_comm.Split(color=1 if world_rank > 0 else MPI.UNDEFINED, key=world_rank - 1 if world_rank > 0 else 0)

    if world_size < 5:
        if world_rank == 0:
            print("Erreur: cette étape nécessite au moins 5 processus (1 affichage + 4 workers).")
        raise SystemExit(1)

    worker_grid = choose_worker_grid(nworkers)
    if worker_grid is None:
        if world_rank == 0:
            print(
                "Erreur: le nombre de workers ne forme pas une grille 2D valide. "
                "Utilisez Stage 02 pour 2 processus, ou Stage 03 avec un nombre de workers factorisable en px x py avec px > 1 et py > 1."
            )
        raise SystemExit(1)

    px, py = worker_grid
    n_cells_per_dir = (args.ni, args.nj, args.nk)

    if world_rank == 0:
        state = load_initial_state(args.filename, n_cells_per_dir)
        _, worker_payloads = build_initial_payloads(state, px, py)
        send_initial_payloads(world_comm, worker_payloads)

        if args.benchmark:
            print(
                f"Benchmark MPI complet de {args.filename} avec dt = {args.dt}, grille {n_cells_per_dir}, "
                f"workers = {nworkers} ({px}x{py}), warmup = {args.warmup}, steps = {args.steps}"
            )
            run_display_rank_benchmark(world_comm, state, steps=args.steps, warmup=args.warmup)
        else:
            print(
                f"Simulation MPI complète de {args.filename} avec dt = {args.dt}, grille {n_cells_per_dir}, "
                f"workers = {nworkers} ({px}x{py})"
            )
            run_display_rank_interactive(world_comm, state, dt=args.dt)
    else:
        init_payload = receive_initial_payload(world_comm)
        if args.benchmark:
            run_worker_rank_benchmark(world_comm, compute_comm, init_payload, dt=args.dt, steps=args.steps, warmup=args.warmup)
        else:
            run_worker_rank_interactive(world_comm, compute_comm, init_payload, dt=args.dt)


if __name__ == "__main__":
    main()
