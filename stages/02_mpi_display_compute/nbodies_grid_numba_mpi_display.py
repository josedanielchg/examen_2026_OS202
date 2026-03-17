# Simulation d'une galaxie à n corps en utilisant une grille spatiale pour accélérer le calcul des forces gravitationnelles.
# Cette variante de Stage 02 sépare le calcul et l'affichage avec MPI :
# rank 0 gère la visualisation, rank 1 conserve l'état physique complet.
import argparse
import numpy as np
import visualizer3d
import time
from mpi4py import MPI
from numba import get_num_threads, njit, prange

# Unités:
# - Distance: année-lumière (ly)
# - Masse: masse solaire (M_sun)
# - Vitesse: année-lumière par an (ly/an)
# - Temps: année

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13

TAG_INIT = 10
TAG_CMD = 11
TAG_POS = 12
TAG_ACK = 13
TAG_STATS = 14

CMD_STEP = "STEP"
CMD_STOP = "STOP"

def generate_star_color(mass : float) -> tuple[int, int, int]:
    """
    Génère une couleur pour une étoile en fonction de sa masse.
    Les étoiles massives sont bleues, les moyennes sont jaunes, les petites sont rouges.
    
    Parameters:
    -----------
    mass : float
        Masse de l'étoile en masses solaires
    
    Returns:
    --------
    color : tuple
        Couleur RGB (R, G, B) avec des valeurs entre 0 et 255
    """
    if mass > 5.0:
        # Étoiles massives: bleu-blanc
        return (150, 180, 255)
    elif mass > 2.0:
        # Étoiles moyennes-massives: blanc
        return (255, 255, 255)
    elif mass >= 1.0:
        # Étoiles comme le Soleil: jaune
        return (255, 255, 200)
    else:
        # Étoiles de faible masse: rouge-orange
        return (255, 150, 100)

@njit(parallel=True)
def compute_cell_mass_and_com(cell_start_indices : np.ndarray, body_indices : np.ndarray,
                              cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                              masses : np.ndarray, positions : np.ndarray):
    for i in prange(cell_masses.shape[0]):
        cell_mass = 0.0
        com_x = 0.0
        com_y = 0.0
        com_z = 0.0
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_x += positions[ibody, 0] * m
            com_y += positions[ibody, 1] * m
            com_z += positions[ibody, 2] * m
        cell_masses[i] = cell_mass
        if cell_mass > 0.0:
            inv_mass = 1.0 / cell_mass
            cell_com_positions[i, 0] = com_x * inv_mass
            cell_com_positions[i, 1] = com_y * inv_mass
            cell_com_positions[i, 2] = com_z * inv_mass
        else:
            cell_com_positions[i, 0] = 0.0
            cell_com_positions[i, 1] = 0.0
            cell_com_positions[i, 2] = 0.0

@njit
def update_stars_in_grid( cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          masses: np.ndarray,
                          positions : np.ndarray, grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    # Réinitialise les compteurs de début des cellules
    cell_start_indices.fill(-1)
    # Compte le nombre de corps dans chaque cellule
    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        # Gère le cas où un corps est exactement sur la borne max   
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        cell_counts[morse_idx] += 1
    # Calcule les indices de début des cellules
    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index # Fin du dernier corps
    # Remplit les indices des corps dans les cellules
    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1]*n_cells[0] + cell_idx[2]*n_cells[0]*n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1
    compute_cell_mass_and_com(cell_start_indices, body_indices, cell_masses, cell_com_positions, masses, positions)

@njit(parallel=True)
def compute_acceleration( positions : np.ndarray, masses : np.ndarray,
                          cell_start_indices : np.ndarray, body_indices : np.ndarray,
                          cell_masses : np.ndarray, cell_com_positions : np.ndarray,
                          grid_min : np.ndarray, grid_max : np.ndarray,
                          cell_size : np.ndarray, n_cells : np.ndarray):
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)
    for ibody in prange(n_bodies):
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        # Parcourt toutes les cellules pour calculer la contribution gravitationnelle
        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy*n_cells[0] + iz*n_cells[0]*n_cells[1]
                    if (abs(ix-cell_idx[0]) > 2) or (abs(iy-cell_idx[1]) > 2) or (abs(iz-cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]    
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                            if distance > 1.E-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[ibody,:] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        # Parcourt les corps dans cette cellule
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx+1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
                                if distance > 1.E-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a[ibody,:] += G * direction[:] * inv_dist3 * masses[jbody]
    return a

@njit(parallel=True)
def integrate_positions(positions : np.ndarray, velocities : np.ndarray, accelerations : np.ndarray, dt : float):
    for ibody in prange(positions.shape[0]):
        positions[ibody, 0] += velocities[ibody, 0] * dt + 0.5 * accelerations[ibody, 0] * dt * dt
        positions[ibody, 1] += velocities[ibody, 1] * dt + 0.5 * accelerations[ibody, 1] * dt * dt
        positions[ibody, 2] += velocities[ibody, 2] * dt + 0.5 * accelerations[ibody, 2] * dt * dt

@njit(parallel=True)
def integrate_velocities(velocities : np.ndarray, accelerations : np.ndarray, accelerations_new : np.ndarray, dt : float):
    for ibody in prange(velocities.shape[0]):
        velocities[ibody, 0] += 0.5 * (accelerations[ibody, 0] + accelerations_new[ibody, 0]) * dt
        velocities[ibody, 1] += 0.5 * (accelerations[ibody, 1] + accelerations_new[ibody, 1]) * dt
        velocities[ibody, 2] += 0.5 * (accelerations[ibody, 2] + accelerations_new[ibody, 2]) * dt

# On crée une grille cartésienne régulière pour diviser l'espace englobant la galaxie en cellules
class SpatialGrid:
    """_summary_
    """
    def __init__(self, positions : np.ndarray, nb_cells_per_dim : tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        # On va stocker les indices des corps dans chaque cellule adéquate
        # Les cellules seront stockées sous une forme morse : indice de la cellule = ix + iy*n_cells_x + iz*n_cells_x*n_cells_y
        # et on gère deux tableaux : un pour le début des indices de chaque cellule, un autre pour les indices des corps
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        # Stockage du centre de masse de chaque cellule et de la masse totale contenue dans chaque cellule
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)
        
    def update_bounds(self, positions : np.ndarray):
        self.min_bounds = np.min(positions, axis=0) - 1.E-6
        self.max_bounds = np.max(positions, axis=0) + 1.E-6
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        
    def update(self, positions : np.ndarray, masses : np.ndarray):
        #self.update_bounds(positions)
        update_stars_in_grid( self.cell_start_indices, self.body_indices,                             
                              self.cell_masses, self.cell_com_positions,
                              masses,
                              positions, self.min_bounds, self.max_bounds,
                              self.cell_size, self.n_cells)

class NBodySystem:
    def __init__(self, filename, ncells_per_dir : tuple[int, int, int] = (10,10,10)):
        positions = []
        velocities = []
        masses    = []
        
        self.max_mass = 0.
        self.box = np.array([[-1.E-6,-1.E-6,-1.E-6],[1.E-6,1.E-6,1.E-6]], dtype=np.float64) # Contient les coins min et max du système
        with open(filename, "r") as fich:
            line = fich.readline() # Récupère la masse, la position et la vitesse sous forme de chaîne
            # Récupère les données numériques pour instancier un corps qu'on rajoute aux corps déjà présents :
            while line:
                data = line.split()
                masses.append(float(data[0]))
                positions.append([float(data[1]), float(data[2]), float(data[3])])
                velocities.append([float(data[4]), float(data[5]), float(data[6])])
                self.max_mass = max(self.max_mass, masses[-1])
                
                for i in range(3):
                    self.box[0][i] = min(self.box[0][i], positions[-1][i]-1.E-6)
                    self.box[1][i] = max(self.box[1][i], positions[-1][i]+1.E-6)
                    
                line = fich.readline()
        
        self.positions  = np.array(positions, dtype=np.float32)
        self.velocities = np.array(velocities, dtype=np.float32)
        self.masses     = np.array(masses, dtype=np.float32)
        self.colors = [generate_star_color(m) for m in masses]
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)
        
    def update_positions(self, dt):
        """Applique la méthode de Verlet vectorisée pour mettre à jour les positions et vitesses des corps."""
        a = compute_acceleration( self.positions, self.masses,
                                  self.grid.cell_start_indices, self.grid.body_indices,
                                  self.grid.cell_masses, self.grid.cell_com_positions,
                                  self.grid.min_bounds, self.grid.max_bounds,
                                  self.grid.cell_size, self.grid.n_cells)
        integrate_positions(self.positions, self.velocities, a, dt)
        self.grid.update(self.positions, self.masses)
        a_new = compute_acceleration( self.positions, self.masses,
                                      self.grid.cell_start_indices, self.grid.body_indices,
                                      self.grid.cell_masses, self.grid.cell_com_positions,
                                      self.grid.min_bounds, self.grid.max_bounds,
                                      self.grid.cell_size, self.grid.n_cells)
        integrate_velocities(self.velocities, a, a_new, dt)

def build_visual_payload(system : NBodySystem) -> dict:
    return {
        "positions": np.array(system.positions, copy=True),
        "colors": np.array(system.colors, dtype=np.float32),
        "luminosities": np.clip(system.masses / system.max_mass, 0.5, 1.0).astype(np.float32),
        "bounds": [
            [system.box[0][0], system.box[1][0]],
            [system.box[0][1], system.box[1][1]],
            [system.box[0][2], system.box[1][2]],
        ],
        "shape": tuple(system.positions.shape),
        "dtype": str(system.positions.dtype),
    }


def build_benchmark_payload(system : NBodySystem) -> dict:
    return {
        "shape": tuple(system.positions.shape),
        "dtype": str(system.positions.dtype),
    }


def mpi_fetch_positions(comm, position_buffer : np.ndarray) -> np.ndarray:
    request = comm.Irecv(position_buffer, source=1, tag=TAG_POS)
    comm.send(CMD_STEP, dest=1, tag=TAG_CMD)
    request.Wait()
    return position_buffer


def run_display_rank_interactive(comm, dt : float):
    init_payload = comm.recv(source=1, tag=TAG_INIT)
    position_buffer = np.empty(init_payload["shape"], dtype=np.dtype(init_payload["dtype"]))

    visualizer = visualizer3d.Visualizer3D(
        init_payload["positions"],
        init_payload["colors"],
        init_payload["luminosities"],
        init_payload["bounds"],
    )

    def updater(_dt : float) -> np.ndarray:
        return mpi_fetch_positions(comm, position_buffer)

    try:
        visualizer.run(updater=updater, dt=dt)
    finally:
        comm.send(CMD_STOP, dest=1, tag=TAG_CMD)
        ack = comm.recv(source=1, tag=TAG_ACK)
        if ack != "ACK":
            raise RuntimeError("Le rank de calcul n'a pas confirmé l'arrêt correctement.")


def run_compute_rank_interactive(comm, filename : str, ncells_per_dir : tuple[int, int, int], dt : float):
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    comm.send(build_visual_payload(system), dest=0, tag=TAG_INIT)

    while True:
        cmd = comm.recv(source=0, tag=TAG_CMD)
        if cmd == CMD_STEP:
            system.update_positions(dt)
            comm.Send([system.positions, MPI.FLOAT], dest=0, tag=TAG_POS)
        elif cmd == CMD_STOP:
            comm.send("ACK", dest=0, tag=TAG_ACK)
            return
        else:
            raise RuntimeError(f"Commande MPI inconnue: {cmd}")


def run_display_rank_benchmark(comm, steps : int, warmup : int):
    init_payload = comm.recv(source=1, tag=TAG_INIT)
    position_buffer = np.empty(init_payload["shape"], dtype=np.dtype(init_payload["dtype"]))
    end_to_end_times = np.empty(steps, dtype=np.float64)

    total_steps = warmup + steps
    for istep in range(total_steps):
        request = comm.Irecv(position_buffer, source=1, tag=TAG_POS)
        t0 = time.perf_counter()
        comm.send(CMD_STEP, dest=1, tag=TAG_CMD)
        request.Wait()
        elapsed = time.perf_counter() - t0
        if istep >= warmup:
            end_to_end_times[istep - warmup] = elapsed

    comm.send(CMD_STOP, dest=1, tag=TAG_CMD)
    stats = comm.recv(source=1, tag=TAG_STATS)
    ack = comm.recv(source=1, tag=TAG_ACK)
    if ack != "ACK":
        raise RuntimeError("Le rank de calcul n'a pas confirmé l'arrêt correctement.")

    end_to_end_total = float(np.sum(end_to_end_times))
    end_to_end_mean = float(np.mean(end_to_end_times))
    end_to_end_std = float(np.std(end_to_end_times))
    rank1_compute_mean = float(stats["rank1_compute_mean_s"])
    rank1_compute_std = float(stats["rank1_compute_std_s"])
    mpi_overhead_mean = end_to_end_mean - rank1_compute_mean

    print(
        "BENCHMARK_MPI "
        f"threads={stats['threads']} "
        f"warmup={warmup} "
        f"steps={steps} "
        f"rank1_compute_ms={rank1_compute_mean * 1000.0:.3f} "
        f"rank1_compute_std_ms={rank1_compute_std * 1000.0:.3f} "
        f"end_to_end_ms={end_to_end_mean * 1000.0:.3f} "
        f"end_to_end_std_ms={end_to_end_std * 1000.0:.3f} "
        f"mpi_overhead_ms={mpi_overhead_mean * 1000.0:.3f} "
        f"total_end_to_end_s={end_to_end_total:.6f}"
    )


def run_compute_rank_benchmark(comm, filename : str, ncells_per_dir : tuple[int, int, int], dt : float, steps : int, warmup : int):
    system = NBodySystem(filename, ncells_per_dir=ncells_per_dir)
    comm.send(build_benchmark_payload(system), dest=0, tag=TAG_INIT)

    compute_times = np.empty(steps, dtype=np.float64)
    step_index = 0

    while True:
        cmd = comm.recv(source=0, tag=TAG_CMD)
        if cmd == CMD_STEP:
            t0 = time.perf_counter()
            system.update_positions(dt)
            elapsed = time.perf_counter() - t0
            if step_index >= warmup:
                compute_times[step_index - warmup] = elapsed
            step_index += 1
            comm.Send([system.positions, MPI.FLOAT], dest=0, tag=TAG_POS)
        elif cmd == CMD_STOP:
            comm.send(
                {
                    "threads": get_num_threads(),
                    "rank1_compute_mean_s": float(np.mean(compute_times)),
                    "rank1_compute_std_s": float(np.std(compute_times)),
                    "rank1_compute_total_s": float(np.sum(compute_times)),
                },
                dest=0,
                tag=TAG_STATS,
            )
            comm.send("ACK", dest=0, tag=TAG_ACK)
            return
        else:
            raise RuntimeError(f"Commande MPI inconnue: {cmd}")

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation N-corps Stage 02: affichage sur rank 0 et calcul sur rank 1 avec MPI.")
    parser.add_argument("filename", nargs="?", default="data/galaxy_1000")
    parser.add_argument("dt", nargs="?", type=float, default=0.001)
    parser.add_argument("ni", nargs="?", type=int, default=20)
    parser.add_argument("nj", nargs="?", type=int, default=20)
    parser.add_argument("nk", nargs="?", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true", help="Exécute un benchmark MPI sans ouvrir la fenêtre SDL/OpenGL.")
    parser.add_argument("--steps", type=int, default=30, help="Nombre de pas mesurés en mode benchmark MPI.")
    parser.add_argument("--warmup", type=int, default=1, help="Nombre de pas non mesurés pour amortir la compilation JIT.")
    return parser.parse_args()

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_cells_per_dir = (args.ni, args.nj, args.nk)

    if size != 2:
        if rank == 0:
            print("Erreur: cette étape MPI doit être exécutée exactement avec 2 processus.")
        raise SystemExit(1)

    if args.benchmark:
        if rank == 0:
            print(
                f"Benchmark MPI de {args.filename} avec dt = {args.dt}, grille {n_cells_per_dir}, "
                f"warmup = {args.warmup}, steps = {args.steps}"
            )
            run_display_rank_benchmark(comm, steps=args.steps, warmup=args.warmup)
        else:
            run_compute_rank_benchmark(
                comm,
                args.filename,
                ncells_per_dir=n_cells_per_dir,
                dt=args.dt,
                steps=args.steps,
                warmup=args.warmup,
            )
    else:
        if rank == 0:
            print(f"Simulation MPI de {args.filename} avec dt = {args.dt} et grille {n_cells_per_dir}")
            run_display_rank_interactive(comm, dt=args.dt)
        else:
            run_compute_rank_interactive(comm, args.filename, ncells_per_dir=n_cells_per_dir, dt=args.dt)

if __name__ == "__main__":
    main()
